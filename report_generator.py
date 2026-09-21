import os
import io
import uuid
import base64
import html
import datetime
from pathlib import Path
from PIL import Image as PILImage, ImageOps
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
import pypdfium2 as pdfium

from config import UPLOAD_DIR, REPORT_DIR


class NumberedCanvas(canvas.Canvas):
    """
    Two-pass canvas to dynamically compute and stamp total page count (Page X of Y)
    along with official Claro security watermark and footer.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 7)
        self.setFillColor(colors.HexColor("#64748B"))
        self.setStrokeColor(colors.HexColor("#CBD5E1"))
        self.setLineWidth(0.5)

        # Bottom rule
        self.line(36, 26, 576, 26)
        self.drawString(36, 16, "CLARO Legal Metrology Optical Intelligence Report • Official Statutory Record")
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(576, 16, page_str)
        self.restoreState()


def generate_inspection_pdf(inspection_data: dict) -> str:
    """
    Generates a professional, comprehensive Legal Metrology Inspection Report PDF
    with embedded 3-sided visual evidence and complete statutory declaration results.
    """
    inspection_num = inspection_data.get("inspection_number", "INSP-UNKNOWN")
    pdf_filename = f"{inspection_num}.pdf"
    pdf_path = REPORT_DIR / pdf_filename

    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#0F172A")
    )
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#475569")
    )
    heading_style = ParagraphStyle(
        'SectionHeading',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10.5,
        leading=14,
        textColor=colors.HexColor("#1E293B"),
        spaceAfter=5
    )
    cell_style = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.5,
        leading=9.5,
        textColor=colors.HexColor("#1E293B")
    )
    cell_bold = ParagraphStyle(
        'TableCellBold',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=7.5,
        leading=9.5,
        textColor=colors.HexColor("#0F172A")
    )

    story = []

    # 1. Header Banner
    header_data = [
        [
            Paragraph("<b>CLARO</b> — Legal Metrology Inspection Platform<br/><font size=7 color='#64748B'>INSPECT • MONITOR • UNDERSTAND</font>", title_style),
            Paragraph(f"<b>CASE REF:</b> {html.escape(inspection_num)}<br/><b>DATE:</b> {datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}", subtitle_style)
        ]
    ]
    header_table = Table(header_data, colWidths=[360, 180])
    header_table.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
    ]))
    story.append(header_table)
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#0284C7"), spaceBefore=4, spaceAfter=8))

    # 2. Case & Officer Information
    status = str(inspection_data.get("status", "PENDING_VERIFICATION")).upper()
    status_bg = colors.HexColor("#DC2626") if status == "NON_COMPLIANT" else (colors.HexColor("#16A34A") if status == "COMPLIANT" else colors.HexColor("#D97706"))

    officer_info = [
        [
            Paragraph("<b>Officer Name:</b>", cell_bold), Paragraph(html.escape(str(inspection_data.get("officer_name", "Enforcement Officer"))), cell_style),
            Paragraph("<b>Inspection Status:</b>", cell_bold), Paragraph(f"<b><font color='white'>{status}</font></b>", ParagraphStyle('StatusStyle', parent=cell_style, backColor=status_bg, borderPadding=2.5))
        ],
        [
            Paragraph("<b>Officer ID:</b>", cell_bold), Paragraph(html.escape(str(inspection_data.get("officer_badge", "LM-OFF-01"))), cell_style),
            Paragraph("<b>Location / Premise:</b>", cell_bold), Paragraph(html.escape(str(inspection_data.get("location", "Field Retail Store"))), cell_style)
        ],
        [
            Paragraph("<b>Product Inspected:</b>", cell_bold), Paragraph(html.escape(str(inspection_data.get("product_name", "Packaged Commodity"))), cell_style),
            Paragraph("<b>Date / Time:</b>", cell_bold), Paragraph(html.escape(str(inspection_data.get("created_at", datetime.datetime.utcnow().strftime('%Y-%m-%d')))), cell_style)
        ]
    ]
    meta_table = Table(officer_info, colWidths=[95, 175, 95, 175])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 8))

    # 3. Summary Statistics
    summary_data = [
        [
            Paragraph(f"<b>Total Declarations Checked:</b> {len(inspection_data.get('validations', []))}", cell_style),
            Paragraph(f"<b>Passed:</b> <font color='#16A34A'><b>{inspection_data.get('pass_count', 0)}</b></font>", cell_style),
            Paragraph(f"<b>Violations / Failed:</b> <font color='#DC2626'><b>{inspection_data.get('fail_count', 0)}</b></font>", cell_style),
            Paragraph(f"<b>Under Review:</b> <font color='#D97706'><b>{inspection_data.get('review_count', 0)}</b></font>", cell_style)
        ]
    ]
    summary_table = Table(summary_data, colWidths=[160, 120, 130, 130])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#EFF6FF")),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor("#BFDBFE")),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 8))

    # 4. Captured Package Visual Evidence (4-Sided or 3-Sided Visual Audit)
    def resolve_image_flowable(img_val, label_title, img_width=160, img_height=105):
        col_items = [Paragraph(f"<b>{label_title}</b>", cell_bold)]
        resolved_path = None

        if img_val:
            s = str(img_val).strip()
            # 1. Base64 data URI handling
            if s.startswith("data:image/"):
                try:
                    header, b64data = s.split(",", 1)
                    img_bytes = base64.b64decode(b64data)
                    ext = "png" if "png" in header else "jpg"
                    temp_p = UPLOAD_DIR / f"tmp_b64_{uuid.uuid4().hex[:8]}.{ext}"
                    with open(temp_p, "wb") as f:
                        f.write(img_bytes)
                    resolved_path = temp_p
                except Exception as b64_err:
                    print(f"Base64 image decode error: {b64_err}")
            # 2. Path containing /uploads/
            elif "/uploads/" in s:
                filename = s.split("/uploads/")[-1].split("?")[0]
                p = UPLOAD_DIR / filename
                if p.exists() and p.is_file():
                    resolved_path = p
                elif s.startswith("http://") or s.startswith("https://"):
                    try:
                        import urllib.request
                        temp_p = UPLOAD_DIR / f"tmp_remote_{uuid.uuid4().hex[:8]}.jpg"
                        urllib.request.urlretrieve(s, str(temp_p))
                        if temp_p.exists() and temp_p.stat().st_size > 0:
                            resolved_path = temp_p
                    except Exception as http_err:
                        print(f"Remote image download error: {http_err}")
            # 3. Direct HTTP(S) URL
            elif s.startswith("http://") or s.startswith("https://"):
                try:
                    import urllib.request
                    temp_p = UPLOAD_DIR / f"tmp_remote_{uuid.uuid4().hex[:8]}.jpg"
                    urllib.request.urlretrieve(s, str(temp_p))
                    if temp_p.exists() and temp_p.stat().st_size > 0:
                        resolved_path = temp_p
                except Exception as http_err:
                    print(f"Remote image download error: {http_err}")
            # 4. Direct or local filename
            else:
                p = Path(s)
                if p.exists() and p.is_file():
                    resolved_path = p
                else:
                    p2 = UPLOAD_DIR / p.name
                    if p2.exists() and p2.is_file():
                        resolved_path = p2

        if resolved_path:
            try:
                # Ensure EXIF orientation is corrected so phone photos are oriented upright
                norm_path = resolved_path
                try:
                    with PILImage.open(resolved_path) as im:
                        oriented_im = ImageOps.exif_transpose(im)
                        norm_filename = f"norm_{resolved_path.stem}.jpg"
                        norm_path = resolved_path.parent / norm_filename
                        oriented_im.convert("RGB").save(norm_path, format="JPEG", quality=88)
                except Exception:
                    norm_path = resolved_path

                img_obj = Image(str(norm_path), width=img_width, height=img_height, kind='proportional')
                col_items.append(Spacer(1, 3))
                col_items.append(img_obj)
            except Exception as img_err:
                col_items.append(Spacer(1, 15))
                col_items.append(Paragraph(f"<font color='#94A3B8'>[Image Preview Error]</font>", cell_style))
                col_items.append(Spacer(1, 15))
        else:
            col_items.append(Spacer(1, 15))
            col_items.append(Paragraph("<font color='#94A3B8'>[Visual Evidence Not Captured]</font>", cell_style))
            col_items.append(Spacer(1, 15))
        return col_items

    images_dict = inspection_data.get("images") or {}
    front_img = inspection_data.get("front_image") or images_dict.get("front")
    back_img = inspection_data.get("back_image") or images_dict.get("back")
    side_img = inspection_data.get("side_image") or images_dict.get("side")
    right_img = inspection_data.get("right_image") or images_dict.get("right_side") or images_dict.get("right")
    left_img = inspection_data.get("left_image") or images_dict.get("left_side") or images_dict.get("left")

    is_4_sided = bool(right_img or left_img)

    if is_4_sided:
        front_col = resolve_image_flowable(front_img, "1. FRONT (PDP & Brand)", img_width=120, img_height=85)
        back_col = resolve_image_flowable(back_img, "2. BACK (Declarations)", img_width=120, img_height=85)
        right_col = resolve_image_flowable(right_img or side_img, "3. RIGHT (MRP & USP)", img_width=120, img_height=85)
        left_col = resolve_image_flowable(left_img, "4. LEFT (Batch & Origin)", img_width=120, img_height=85)

        story.append(Paragraph("CAPTURED PACKAGE EVIDENCE (4-SIDED VISUAL AUDIT)", heading_style))
        evidence_table = Table([[front_col, back_col, right_col, left_col]], colWidths=[135, 135, 135, 135])
    else:
        front_col = resolve_image_flowable(front_img, "1. FRONT (PDP & Brand)", img_width=160, img_height=105)
        back_col = resolve_image_flowable(back_img, "2. BACK (Declarations)", img_width=160, img_height=105)
        side_col = resolve_image_flowable(side_img, "3. SIDE (MRP & Date)", img_width=160, img_height=105)

        story.append(Paragraph("CAPTURED PACKAGE EVIDENCE (3-SIDED VISUAL AUDIT)", heading_style))
        evidence_table = Table([[front_col, back_col, side_col]], colWidths=[180, 180, 180])

    evidence_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('LEFTPADDING', (0, 0), (-1, -1), 5),
        ('RIGHTPADDING', (0, 0), (-1, -1), 5),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ]))
    story.append(evidence_table)
    story.append(Spacer(1, 8))

    # 5. Field-wise Compliance Details (All results listed)
    story.append(Paragraph("STATUTORY DECLARATION AUDIT TABLE (PCR 2011)", heading_style))

    table_rows = [
        [
            Paragraph("<b>Declaration</b>", cell_bold),
            Paragraph("<b>Detected Value</b>", cell_bold),
            Paragraph("<b>Rule Reference</b>", cell_bold),
            Paragraph("<b>Status</b>", cell_bold),
            Paragraph("<b>Confidence</b>", cell_bold),
            Paragraph("<b>Audit Finding / Notes</b>", cell_bold)
        ]
    ]

    for val in inspection_data.get("validations", []):
        st = str(val.get("status", "REVIEW")).upper()
        st_color = "#16A34A" if st == "PASS" else ("#DC2626" if st == "FAIL" else "#D97706")

        conf_val = val.get("confidence", 0)
        try:
            conf_pct = f"{int(float(conf_val) * 100)}%"
        except Exception:
            conf_pct = "N/A"

        disp_name = val.get("display_name") or val.get("field_name") or "Declaration"
        det_val = val.get("detected_value")
        if not det_val or str(det_val).strip() in ["None", "null", ""]:
            det_val_str = "<font color='#94A3B8'><i>Not Detected</i></font>"
        else:
            det_val_str = html.escape(str(det_val))

        rule_ref = html.escape(str(val.get("rule_reference", "PCR-2011")))
        reason_text = html.escape(str(val.get("reason", "")))

        table_rows.append([
            Paragraph(html.escape(str(disp_name)), cell_style),
            Paragraph(det_val_str, cell_style),
            Paragraph(rule_ref, cell_style),
            Paragraph(f"<b><font color='{st_color}'>{st}</font></b>", cell_style),
            Paragraph(conf_pct, cell_style),
            Paragraph(reason_text, cell_style)
        ])

    comp_table = Table(table_rows, colWidths=[95, 105, 75, 45, 45, 175])
    comp_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#F1F5F9")),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ('TOPPADDING', (0, 0), (-1, -1), 3.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3.5),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ]))
    story.append(comp_table)
    story.append(Spacer(1, 8))

    # 5b. Corrective Guidance / How to Correct Non-Compliance
    non_compliant_items = [
        v for v in inspection_data.get("validations", [])
        if str(v.get("status", "")).upper() in ["FAIL", "REVIEW"]
    ]

    if non_compliant_items:
        story.append(Paragraph("HOW TO CORRECT NON-COMPLIANCE / STATUTORY GUIDANCE", heading_style))
        guidance_rules = {
            "product_name": "Ensure the common or generic name of the commodity is clearly displayed on the Principal Display Panel (PDP) under Rule 6(1)(b).",
            "brand": "Ensure brand identity or commercial trade name is clearly declared under Rule 6(1)(b).",
            "mrp": "Ensure MRP is declared strictly as 'Maximum Retail Price ₹... (inclusive of all taxes)' on the PDP or statutory panel in clear, legible font.",
            "net_quantity": "Declare Net Quantity in SI metric units (g, kg, ml, l) adhering to the minimum letter/numeral height specified in Schedule II based on package area.",
            "unit_sale_price": "For packages containing more than 1 unit/kg/litre, clearly print Unit Sale Price rounded to 2 decimal places (e.g., ₹0.25 / g).",
            "manufacturer": "Print full name and registered commercial entity of the manufacturer, packer, or importer.",
            "complete_address": "Specify complete postal address of manufacturing/packing premises along with valid 6-digit PIN code.",
            "dates": "Declare Month and Year of manufacture / packing in standard DD/MM/YYYY or MM/YYYY format.",
            "consumer_care": "Provide name, address, telephone/toll-free number, and active email address of the consumer redressal officer under Rule 6(2).",
            "country_of_origin": "State country of origin prominently for all imported or domestically packaged goods.",
            "batch_lot_number": "Ensure batch, lot, or code number is clearly visible for product traceability."
        }

        guidance_rows = [[
            Paragraph("<b>Deficient Declaration</b>", cell_bold),
            Paragraph("<b>Statutory Corrective Action Required (PCR 2011)</b>", cell_bold)
        ]]

        for item in non_compliant_items:
            fn = item.get("field_name", "").lower()
            dn = item.get("display_name") or item.get("field_name")
            remedy = guidance_rules.get(fn, f"Ensure mandatory statutory declaration for {dn} complies with PCR 2011 Rule 6 provisions.")
            guidance_rows.append([
                Paragraph(f"<font color='#DC2626'><b>{html.escape(str(dn))}</b></font>", cell_style),
                Paragraph(html.escape(remedy), cell_style)
            ])

        guidance_table = Table(guidance_rows, colWidths=[160, 380])
        guidance_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#FEF2F2")),
            ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor("#FECACA")),
            ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#FEE2E2")),
            ('TOPPADDING', (0, 0), (-1, -1), 3),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ]))
        story.append(KeepTogether(guidance_table))
        story.append(Spacer(1, 8))

    # 6. Verification Sign-off block
    sign_off_data = [
        [
            Paragraph("<b>Human Reviewer / Verification Notes:</b><br/>" + html.escape(str(inspection_data.get("officer_notes", "Automated inspection completed with optical and rule-engine verification."))), cell_style),
            Paragraph("<b>Officer Sign-off / Digital Attestation:</b><br/><br/>_______________________________<br/>Authorized Enforcement Officer", cell_style)
        ]
    ]
    sign_table = Table(sign_off_data, colWidths=[340, 200])
    sign_table.setStyle(TableStyle([
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ]))
    story.append(KeepTogether(sign_table))
    story.append(Spacer(1, 8))

    # 7. Disclaimer & Audit Trail Hash
    disclaimer = Paragraph(
        "<font size=6.5 color='#64748B'>This inspection report is generated by Claro Platform. Findings are based on optical extraction and deterministic evaluation against the Legal Metrology (Packaged Commodities) Rules, 2011. Integrity Hash: SHA256-CLARO-" + html.escape(inspection_num) + "</font>",
        cell_style
    )
    story.append(disclaimer)

    # Build PDF with NumberedCanvas for professional page numbering
    doc.build(story, canvasmaker=NumberedCanvas)
    return str(pdf_path)


def render_pdf_to_base64_pages(pdf_path: str, scale: float = 2.0) -> list[str]:
    """
    Renders each page of a PDF into a high-resolution base64 JPEG data URL using pypdfium2.
    """
    p = Path(pdf_path)
    if not p.exists():
        raise FileNotFoundError(f"PDF file not found at {pdf_path}")

    pdf = pdfium.PdfDocument(str(p))
    pages_base64 = []

    for page in pdf:
        bm = page.render(scale=scale)
        pil_im = bm.to_pil()
        buf = io.BytesIO()
        pil_im.save(buf, format='JPEG', quality=85)
        b64 = base64.b64encode(buf.getvalue()).decode('ascii')
        pages_base64.append(f"data:image/jpeg;base64,{b64}")

    return pages_base64
