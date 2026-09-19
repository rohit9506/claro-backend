import re
import secrets
import json
import datetime
from typing import Optional, List
from pydantic import BaseModel, EmailStr
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy import func

from database import get_db
from models import User, Product, Inspection, FieldValidation, Rule, Verification, AuditLog
from auth_deps import get_current_user, require_role, log_audit
from security import (
    hash_password, validate_email_address, evaluate_password_strength, generate_random_password
)

router = APIRouter(prefix="/admin", tags=["Admin Monitoring & Management"])

def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)

# Schemas
class CreateOfficerRequest(BaseModel):
    full_name: str
    email: str
    officer_id: str
    credential_method: Optional[str] = "MANUAL"  # "GENERATE" or "MANUAL"
    username: Optional[str] = None
    password: Optional[str] = None
    confirm_password: Optional[str] = None
    department: Optional[str] = "Legal Metrology Enforcement Wing"
    mobile_number: Optional[str] = None

class AdminResetPasswordRequest(BaseModel):
    method: Optional[str] = "GENERATE"  # "GENERATE" or "MANUAL"
    new_password: Optional[str] = None
    confirm_password: Optional[str] = None

class UpdateOfficerRequest(BaseModel):
    full_name: Optional[str] = None
    department: Optional[str] = None
    mobile_number: Optional[str] = None
    is_active: Optional[bool] = None
    new_password: Optional[str] = None

class RuleRequest(BaseModel):
    rule_code: str
    rule_name: str
    declaration_checked: str
    requirement: str
    applicable_category: Optional[str] = "All Packaged Commodities"
    validation_criteria: str
    legal_source: Optional[str] = "Legal Metrology (Packaged Commodities) Rules, 2011"
    is_active: Optional[bool] = True

class AdjudicateRequest(BaseModel):
    field_validation_id: Optional[int] = None
    decision: str  # CONFIRM_COMPLIANT, MARK_NON_COMPLIANT, REQUEST_RESCAN
    comment: Optional[str] = "Adjudicated by Legal Metrology Administrator"
    corrected_value: Optional[str] = None

# 1. Executive Dashboard KPIs & Real Analytics
@router.get("/dashboard-stats")
def get_admin_dashboard_stats(
    current_user: User = Depends(require_role("ROLE_ADMIN")),
    db: Session = Depends(get_db)
):
    total_officers = db.query(User).filter(User.role == "ROLE_OFFICER").count()
    total_users = db.query(User).filter(User.role == "ROLE_USER").count()
    total_products = db.query(Product).count()
    total_inspections = db.query(Inspection).count()
    
    compliant_count = db.query(Inspection).filter(Inspection.status == "COMPLIANT").count()
    non_compliant_count = db.query(Inspection).filter(Inspection.status == "NON_COMPLIANT").count()
    pending_verif_count = db.query(Inspection).filter(Inspection.status == "PENDING_VERIFICATION").count()

    # Violation category counts from FieldValidation
    violations_by_field = (
        db.query(FieldValidation.field_name, func.count(FieldValidation.id))
        .filter(FieldValidation.status == "FAIL")
        .group_by(FieldValidation.field_name)
        .all()
    )
    violation_stats = [
        {"field": row[0].replace("_", " ").title(), "count": row[1]}
        for row in violations_by_field
    ]

    # Recent inspections
    recent_inspections = (
        db.query(Inspection)
        .order_by(Inspection.created_at.desc())
        .limit(5)
        .all()
    )

    return {
        "kpis": {
            "total_officers": total_officers,
            "total_users": total_users,
            "total_products": total_products,
            "total_inspections": total_inspections,
            "compliant_count": compliant_count,
            "non_compliant_count": non_compliant_count,
            "pending_verifications": pending_verif_count
        },
        "violation_stats": violation_stats,
        "recent_inspections": [
            {
                "id": insp.id,
                "inspection_number": insp.inspection_number,
                "product_name": insp.product_name,
                "status": insp.status,
                "created_at": insp.created_at.strftime("%Y-%m-%d %H:%M") if insp.created_at else None
            }
            for insp in recent_inspections
        ]
    }

# 2. Officer Management
@router.get("/officers")
def list_officers(
    current_user: User = Depends(require_role("ROLE_ADMIN")),
    db: Session = Depends(get_db)
):
    officers = db.query(User).filter(User.role == "ROLE_OFFICER").order_by(User.created_at.desc()).all()
    return [
        {
            "id": off.id,
            "full_name": off.full_name,
            "username": off.username,
            "email": off.email,
            "officer_id": off.officer_id,
            "department": off.department,
            "mobile_number": off.mobile_number,
            "is_active": off.is_active,
            "inspections_count": len(off.inspections),
            "created_at": off.created_at.strftime("%Y-%m-%d") if off.created_at else None
        }
        for off in officers
    ]

@router.post("/officers")
def create_officer(
    req: CreateOfficerRequest,
    current_user: User = Depends(require_role("ROLE_ADMIN")),
    db: Session = Depends(get_db)
):
    # Validate email
    is_valid_email, norm_email = validate_email_address(req.email)
    if not is_valid_email:
        raise HTTPException(status_code=400, detail=f"Invalid email: {norm_email}")

    clean_officer_id = req.officer_id.strip()
    if db.query(User).filter(User.officer_id == clean_officer_id).first():
        raise HTTPException(status_code=400, detail="Officer Badge / ID already registered.")

    if db.query(User).filter(User.email == norm_email).first():
        raise HTTPException(status_code=400, detail="An account with this email already exists.")

    # Determine Username & Password based on Option A (GENERATE) or Option B (MANUAL)
    if req.credential_method == "GENERATE":
        # Generate clean username based on name or officer id
        clean_name = re.sub(r'[^a-zA-Z0-9]', '_', req.full_name.strip().lower())
        rand_suffix = secrets.randbelow(900) + 100
        clean_u = f"officer_{clean_name[:12]}_{rand_suffix}"
        # Ensure username uniqueness
        while db.query(User).filter(User.username == clean_u).first():
            rand_suffix = secrets.randbelow(900) + 100
            clean_u = f"officer_{clean_name[:12]}_{rand_suffix}"
            
        plain_password = generate_random_password(16)
    else:
        # Option B: Manual credentials
        if not req.username:
            raise HTTPException(status_code=400, detail="Username is required for manual credential creation.")
        clean_u = req.username.strip().lower()
        if db.query(User).filter(User.username == clean_u).first():
            raise HTTPException(status_code=400, detail="Officer username already exists.")

        if not req.password:
            raise HTTPException(status_code=400, detail="Password is required.")
        if req.confirm_password and req.password != req.confirm_password:
            raise HTTPException(status_code=400, detail="Passwords do not match.")

        pwd_eval = evaluate_password_strength(req.password)
        if not pwd_eval["is_valid"]:
            raise HTTPException(status_code=400, detail=pwd_eval["errors"][0])
        plain_password = req.password

    officer = User(
        full_name=req.full_name.strip(),
        username=clean_u,
        email=norm_email,
        password_hash=hash_password(plain_password),
        role="ROLE_OFFICER",
        officer_id=clean_officer_id,
        department=req.department.strip() if req.department else "Legal Metrology Enforcement Wing",
        mobile_number=req.mobile_number.strip() if req.mobile_number else None,
        is_active=True,
        is_temporary_password=True,  # Forces officer to update temporary password on first login
        email_verified=True,  # Admin-created officers are pre-verified
        email_verified_at=utc_now()
    )
    db.add(officer)
    db.commit()
    db.refresh(officer)

    log_audit(
        db, actor_id=current_user.id, actor_name=current_user.full_name,
        actor_role=current_user.role, action="OFFICER_CREATED",
        target_type="OFFICER", target_id=str(officer.id),
        details={"officer_id": clean_officer_id, "username": clean_u, "method": req.credential_method}
    )

    return {
        "success": True,
        "message": f"Officer account '{clean_u}' created successfully.",
        "officer_id": officer.id,
        "credentials": {
            "full_name": officer.full_name,
            "officer_id": officer.officer_id,
            "username": officer.username,
            "temporary_password": plain_password,
            "is_temporary": True,
            "department": officer.department
        }
    }

@router.delete("/officers/{officer_id}")
def delete_officer(
    officer_id: int,
    current_user: User = Depends(require_role("ROLE_ADMIN")),
    db: Session = Depends(get_db)
):
    """
    Permanently removes an officer account.
    Safely dissociates inspections so historic audits remain intact with audit trail,
    and removes officer access credentials.
    """
    officer = db.query(User).filter(User.id == officer_id, User.role == "ROLE_OFFICER").first()
    if not officer:
        raise HTTPException(status_code=404, detail="Officer account not found.")

    officer_name = officer.full_name
    officer_badge = officer.officer_id or "N/A"
    officer_username = officer.username

    # 1. Nullify officer_id on inspections so historical records are not lost
    db.query(Inspection).filter(Inspection.officer_id == officer_id).update({"officer_id": None}, synchronize_session=False)

    # 2. Delete any verifications authored by this officer
    db.query(Verification).filter(Verification.reviewer_id == officer_id).delete(synchronize_session=False)

    # 3. Delete the user
    db.delete(officer)
    db.commit()

    log_audit(
        db, actor_id=current_user.id, actor_name=current_user.full_name,
        actor_role=current_user.role, action="OFFICER_DELETED",
        target_type="OFFICER", target_id=str(officer_id),
        details={"name": officer_name, "badge": officer_badge, "username": officer_username}
    )

    return {
        "success": True,
        "message": f"Officer '{officer_name}' ({officer_badge}) removed successfully."
    }

# Dedicated Admin Credential Management & Password Reset
@router.get("/credentials")
def list_credentials(
    current_user: User = Depends(require_role("ROLE_ADMIN")),
    db: Session = Depends(get_db)
):
    """Retrieve all officer and user accounts for credential oversight."""
    accounts = db.query(User).filter(User.role.in_(["ROLE_OFFICER", "ROLE_USER"])).order_by(User.created_at.desc()).all()
    return [
        {
            "id": acc.id,
            "full_name": acc.full_name,
            "username": acc.username,
            "email": acc.email,
            "role": acc.role,
            "officer_id": acc.officer_id,
            "department": acc.department,
            "is_active": acc.is_active,
            "is_temporary_password": acc.is_temporary_password,
            "email_verified": acc.email_verified,
            "created_at": acc.created_at.strftime("%Y-%m-%d") if acc.created_at else None,
            "last_login_at": acc.last_login_at.strftime("%Y-%m-%d %H:%M") if acc.last_login_at else "Never"
        }
        for acc in accounts
    ]

@router.post("/accounts/{user_id}/reset-password")
def admin_reset_account_password(
    user_id: int,
    req: AdminResetPasswordRequest,
    current_user: User = Depends(require_role("ROLE_ADMIN")),
    db: Session = Depends(get_db)
):
    """Admin resets an account password with either GENERATE or MANUAL method."""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="Account not found.")

    if req.method == "GENERATE":
        plain_password = generate_random_password(16)
    else:
        if not req.new_password:
            raise HTTPException(status_code=400, detail="New password is required.")
        if req.confirm_password and req.new_password != req.confirm_password:
            raise HTTPException(status_code=400, detail="Passwords do not match.")
        pwd_eval = evaluate_password_strength(req.new_password)
        if not pwd_eval["is_valid"]:
            raise HTTPException(status_code=400, detail=pwd_eval["errors"][0])
        plain_password = req.new_password

    # Update hash and mark temporary
    user.password_hash = hash_password(plain_password)
    user.is_temporary_password = True
    user.updated_at = utc_now()
    db.commit()

    log_audit(
        db, actor_id=current_user.id, actor_name=current_user.full_name,
        actor_role=current_user.role, action="ADMIN_RESET_PASSWORD",
        target_type="USER", target_id=str(user.id),
        details={"target_username": user.username, "target_role": user.role, "method": req.method}
    )

    return {
        "success": True,
        "message": f"Password reset successfully for {user.username}.",
        "credentials": {
            "full_name": user.full_name,
            "username": user.username,
            "officer_id": user.officer_id,
            "temporary_password": plain_password,
            "is_temporary": True
        }
    }

@router.put("/officers/{officer_id}")
def update_officer(
    officer_id: int,
    req: UpdateOfficerRequest,
    current_user: User = Depends(require_role("ROLE_ADMIN")),
    db: Session = Depends(get_db)
):
    officer = db.query(User).filter(User.id == officer_id, User.role == "ROLE_OFFICER").first()
    if not officer:
        raise HTTPException(status_code=404, detail="Officer not found.")

    if req.full_name is not None:
        officer.full_name = req.full_name.strip()
    if req.department is not None:
        officer.department = req.department.strip()
    if req.mobile_number is not None:
        officer.mobile_number = req.mobile_number.strip()
    if req.is_active is not None:
        officer.is_active = req.is_active
    if req.new_password:
        pwd_eval = evaluate_password_strength(req.new_password)
        if not pwd_eval["is_valid"]:
            raise HTTPException(status_code=400, detail=pwd_eval["errors"][0])
        officer.password_hash = hash_password(req.new_password)
        officer.is_temporary_password = True

    officer.updated_at = utc_now()
    db.commit()

    log_audit(
        db, actor_id=current_user.id, actor_name=current_user.full_name,
        actor_role=current_user.role, action="OFFICER_UPDATED",
        target_type="OFFICER", target_id=str(officer.id)
    )

    return {"success": True, "message": "Officer account updated successfully."}

# 3. User Management
@router.get("/users")
def list_users(
    current_user: User = Depends(require_role("ROLE_ADMIN")),
    db: Session = Depends(get_db)
):
    users = db.query(User).filter(User.role == "ROLE_USER").order_by(User.created_at.desc()).all()
    return [
        {
            "id": u.id,
            "full_name": u.full_name,
            "username": u.username,
            "email": u.email,
            "is_active": u.is_active,
            "email_verified": u.email_verified,
            "reviews_count": len(u.reviews),
            "saved_count": len(u.saved_products),
            "created_at": u.created_at.strftime("%Y-%m-%d") if u.created_at else None
        }
        for u in users
    ]

# 4. Legal Metrology Rules Management
@router.get("/rules")
def get_rules(
    current_user: User = Depends(require_role("ROLE_ADMIN", "ROLE_OFFICER")),
    db: Session = Depends(get_db)
):
    rules = db.query(Rule).order_by(Rule.id.asc()).all()
    return rules

@router.post("/rules")
def create_rule(
    req: RuleRequest,
    current_user: User = Depends(require_role("ROLE_ADMIN")),
    db: Session = Depends(get_db)
):
    existing = db.query(Rule).filter(Rule.rule_code == req.rule_code.strip()).first()
    if existing:
        raise HTTPException(status_code=400, detail="Rule with this rule code already exists.")

    rule = Rule(
        rule_code=req.rule_code.strip(),
        rule_name=req.rule_name.strip(),
        declaration_checked=req.declaration_checked.strip(),
        requirement=req.requirement.strip(),
        applicable_category=req.applicable_category.strip() if req.applicable_category else "All Packaged Commodities",
        validation_criteria=req.validation_criteria.strip(),
        legal_source=req.legal_source.strip() if req.legal_source else "Legal Metrology (Packaged Commodities) Rules, 2011",
        is_active=req.is_active
    )
    db.add(rule)
    db.commit()
    db.refresh(rule)

    log_audit(
        db, actor_id=current_user.id, actor_name=current_user.full_name,
        actor_role=current_user.role, action="RULE_CREATED",
        target_type="RULE", target_id=str(rule.id),
        details={"rule_code": rule.rule_code}
    )

    return {"success": True, "message": "Rule created successfully.", "rule": rule}

@router.put("/rules/{rule_id}")
def update_rule(
    rule_id: int,
    req: RuleRequest,
    current_user: User = Depends(require_role("ROLE_ADMIN")),
    db: Session = Depends(get_db)
):
    rule = db.query(Rule).filter(Rule.id == rule_id).first()
    if not rule:
        raise HTTPException(status_code=404, detail="Rule not found.")

    rule.rule_name = req.rule_name.strip()
    rule.declaration_checked = req.declaration_checked.strip()
    rule.requirement = req.requirement.strip()
    rule.validation_criteria = req.validation_criteria.strip()
    rule.applicable_category = req.applicable_category.strip()
    rule.is_active = req.is_active
    rule.updated_at = utc_now()
    db.commit()

    log_audit(
        db, actor_id=current_user.id, actor_name=current_user.full_name,
        actor_role=current_user.role, action="RULE_MODIFIED",
        target_type="RULE", target_id=str(rule.id)
    )

    return {"success": True, "message": "Rule updated successfully."}

# 5. Human Verification Center Queue & Adjudication
@router.get("/verifications")
def get_verification_queue(
    current_user: User = Depends(require_role("ROLE_ADMIN", "ROLE_OFFICER")),
    db: Session = Depends(get_db)
):
    pending_inspections = (
        db.query(Inspection)
        .filter(Inspection.status == "PENDING_VERIFICATION")
        .order_by(Inspection.created_at.desc())
        .all()
    )

    result = []
    for insp in pending_inspections:
        review_fields = [
            {
                "id": v.id,
                "field_name": v.field_name,
                "detected_value": v.detected_value,
                "requirement_summary": v.requirement_summary,
                "rule_reference": v.rule_reference,
                "status": v.status,
                "confidence": v.confidence,
                "reason": v.reason,
                "image_side": v.image_side,
                "bbox_norm": json.loads(v.bbox_json) if v.bbox_json else [0.1, 0.1, 0.3, 0.9]
            }
            for v in insp.validations
            if v.status == "REVIEW"
        ]

        result.append({
            "inspection_id": insp.id,
            "inspection_number": insp.inspection_number,
            "product_name": insp.product_name,
            "location": insp.location,
            "created_at": insp.created_at.strftime("%Y-%m-%d %H:%M") if insp.created_at else None,
            "officer_name": insp.officer.full_name if insp.officer else "Enforcement Officer",
            "front_image": insp.front_image,
            "back_image": insp.back_image,
            "side_image": insp.side_image,
            "review_fields": review_fields
        })

    return result

@router.post("/verifications/{inspection_id}/adjudicate")
def adjudicate_verification(
    inspection_id: int,
    req: AdjudicateRequest,
    current_user: User = Depends(require_role("ROLE_ADMIN", "ROLE_OFFICER")),
    db: Session = Depends(get_db)
):
    insp = db.query(Inspection).filter(Inspection.id == inspection_id).first()
    if not insp:
        raise HTTPException(status_code=404, detail="Inspection record not found.")

    if req.field_validation_id:
        fv = db.query(FieldValidation).filter(
            FieldValidation.id == req.field_validation_id,
            FieldValidation.inspection_id == inspection_id
        ).first()
        if fv:
            prev_status = fv.status
            new_status = "PASS" if req.decision == "CONFIRM_COMPLIANT" else ("FAIL" if req.decision == "MARK_NON_COMPLIANT" else "REVIEW")
            fv.status = new_status
            if req.corrected_value is not None and req.corrected_value.strip():
                fv.detected_value = req.corrected_value.strip()
            fv.reason += f" [Verified by {current_user.full_name}: {req.comment}]"

            # Create Verification log
            v_log = Verification(
                inspection_id=inspection_id,
                field_validation_id=fv.id,
                reviewer_id=current_user.id,
                reviewer_name=current_user.full_name,
                decision=req.decision,
                previous_status=prev_status,
                comment=req.comment,
                created_at=utc_now()
            )
            db.add(v_log)
    else:
        # Whole-case adjudication: resolve all remaining REVIEW fields
        for fv in insp.validations:
            if fv.status == "REVIEW":
                prev_status = fv.status
                new_status = "PASS" if req.decision == "CONFIRM_COMPLIANT" else ("FAIL" if req.decision == "MARK_NON_COMPLIANT" else "REVIEW")
                fv.status = new_status
                fv.reason += f" [Case verified by {current_user.full_name}: {req.comment}]"
                v_log = Verification(
                    inspection_id=inspection_id,
                    field_validation_id=fv.id,
                    reviewer_id=current_user.id,
                    reviewer_name=current_user.full_name,
                    decision=req.decision,
                    previous_status=prev_status,
                    comment=req.comment,
                    created_at=utc_now()
                )
                db.add(v_log)

    # Re-evaluate overall inspection status
    validations = insp.validations
    has_review = any(v.status == "REVIEW" for v in validations)
    has_fail = any(v.status == "FAIL" for v in validations)

    if has_review:
        insp.status = "PENDING_VERIFICATION"
    elif has_fail:
        insp.status = "NON_COMPLIANT"
    else:
        insp.status = "COMPLIANT"

    insp.pass_count = sum(1 for v in validations if v.status == "PASS")
    insp.fail_count = sum(1 for v in validations if v.status == "FAIL")
    insp.review_count = sum(1 for v in validations if v.status == "REVIEW")
    insp.finalized_at = utc_now() if insp.status != "PENDING_VERIFICATION" else None
    insp.updated_at = utc_now()
    db.commit()

    log_audit(
        db, actor_id=current_user.id, actor_name=current_user.full_name,
        actor_role=current_user.role, action="VERIFICATION_ADJUDICATED",
        target_type="INSPECTION", target_id=str(insp.id),
        details={"decision": req.decision, "final_status": insp.status}
    )

    return {
        "success": True,
        "message": f"Verification decision applied. Inspection status is now {insp.status}.",
        "new_status": insp.status
    }

# 6. Audit Trail
@router.get("/audit-logs")
def get_audit_logs(
    limit: int = 50,
    current_user: User = Depends(require_role("ROLE_ADMIN")),
    db: Session = Depends(get_db)
):
    logs = db.query(AuditLog).order_by(AuditLog.timestamp.desc()).limit(limit).all()
    return [
        {
            "id": l.id,
            "actor_name": l.actor_name,
            "actor_role": l.actor_role,
            "action": l.action,
            "target_type": l.target_type,
            "target_id": l.target_id,
            "details": json.loads(l.details_json) if l.details_json else None,
            "timestamp": l.timestamp.strftime("%Y-%m-%d %H:%M:%S") if l.timestamp else None
        }
        for l in logs
    ]
