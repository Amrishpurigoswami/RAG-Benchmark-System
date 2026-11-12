"""
Generate 7+ test PDFs for the RAG Benchmark System + questions.txt.

Usage:
    python scratch/create_test_docs.py
"""

import sys
import subprocess
from pathlib import Path

DATA_DIR = Path("data")
DOCS_DIR = DATA_DIR  # alias


def _install_reportlab():
    """Install reportlab if missing."""
    try:
        import reportlab
        return True
    except ImportError:
        print("Installing reportlab...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "reportlab"])
        print("Done.\n")
        return True


def _imports():
    """Return (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, A4, styles, colors)."""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
    from reportlab.lib import colors
    return SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, A4, getSampleStyleSheet(), colors


def _mkdoc(path_name, title, parts):
    """Create a PDF from structured parts.

    parts: list of items
      - str -> paragraph body
      - (str, str) -> heading, body
      - [data, colWidths, extra_styles] -> table
    """
    p = DOCS_DIR / path_name
    if p.is_file():
        print(f"  EXISTS: {p.name}")
        return

    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, A4, styles, colors = _imports()
    d = SimpleDocTemplate(str(p), pagesize=A4)
    el = [Paragraph(title, styles["Title"]), Spacer(1, 12)]

    for part in parts:
        if isinstance(part, str):
            el.append(Paragraph(part, styles["Normal"]))
            el.append(Spacer(1, 8))
        elif isinstance(part, tuple) and len(part) == 2:
            hdr, body = part
            el.append(Paragraph(hdr, styles["Heading2"]))
            el.append(Spacer(1, 6))
            el.append(Paragraph(body, styles["Normal"]))
            el.append(Spacer(1, 8))
        elif isinstance(part, list) and len(part) >= 2:
            data, widths = part[0], part[1]
            extra = part[2] if len(part) > 2 else []
            t = Table(data, colWidths=widths)
            base = [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2E5090")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
            ]
            t.setStyle(TableStyle(base + extra))
            el.append(Spacer(1, 6))
            el.append(t)
            el.append(Spacer(1, 8))

    d.build(el)
    print(f"  CREATED: {p.name}")


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    _install_reportlab()

    print("=" * 60)
    print("  Generating Test PDFs")
    print("=" * 60)
    print()

    # --- 1. hemant_story.pdf ---
    _mkdoc("hemant_story.pdf", "Employment Contract", [
        ("1. Employee Details",
         "Employee Name: Hemant Sharma<br/>"
         "Employee ID: EMP-2025-0042<br/>"
         "Designation: Senior Software Engineer<br/>"
         "Department: Engineering<br/>"
         "Date of Joining: 15th January 2025<br/>"
         "Reporting Manager: Mr. Rajesh Kumar (VP Engineering)"),
        ("2. Compensation",
         "Monthly Salary (CTC): ₹1,25,000<br/>"
         "Annual Bonus: ₹2,00,000 (eligible after 6 months)<br/>"
         "Stock Options: 50 ESOPs vesting over 4 years"),
        ("4. Bonus Clause (Clause 6.2)",
         "Clause 6.2(e): The annual bonus may be withheld if the employee:<br/>"
         "- Resigns before bonus payout date<br/>"
         "- Has any active disciplinary proceedings<br/>"
         "- Fails to meet performance criteria<br/>"
         "- Violates company policies or code of conduct"),
        ("6.2 Bonus Withholding Provisions",
         "(a) Bonus shall be paid within 30 days of end of financial year.<br/>"
         "(b) Employees must be in active employment on payout date.<br/>"
         "(c) Resignation before payout date voids bonus entitlement.<br/>"
         "(d) Performance below expectations may result in reduced bonus.<br/>"
         "(e) Bonus may be fully withheld if disciplinary action is pending.<br/>"
         "(f) Company reserves right to claw back bonus paid in case of fraud."),
        ("Salary Structure", ""),
        [[
            ["Component", "Monthly (INR)", "Annual (INR)"],
            ["Basic Pay", "50,000", "6,00,000"],
            ["HRA", "25,000", "3,00,000"],
            ["Special Allowance", "20,000", "2,40,000"],
            ["LTA", "5,000", "60,000"],
            ["Medical Reimbursement", "2,000", "24,000"],
            ["PF (Employer)", "5,000", "60,000"],
            ["Performance Bonus (Accrued)", "16,667", "2,00,000"],
            ["Total", "1,25,000", "15,00,000"],
        ], [180, 110, 110],
         [("BACKGROUND", (0, -1), (-1, -1), (0.91, 0.91, 0.91)),
          ("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold")]],
    ])

    # --- 2. medical_insurance_policy.pdf ---
    _mkdoc("medical_insurance_policy.pdf", "Group Medical Insurance Policy", [
        ("Policy Details",
         "Policy Number: MED-2025-7890<br/>"
         "Issued to: TechCorp India Pvt. Ltd.<br/>"
         "Insurer: SafeHealth Insurance Co.<br/>"
         "Policy Period: 01-Jan-2025 to 31-Dec-2025"),
        ("Coverage Details",
         "In-patient hospitalization: ₹5,00,000 per annum<br/>"
         "Out-patient consultation: ₹25,000 per annum<br/>"
         "Dental coverage: ₹15,000 per annum<br/>"
         "Maternity coverage: ₹50,000<br/>"
         "Critical Illness: ₹10,00,000 (one-time)"),
        ("Claims Procedure",
         "To file a claim, submit within 7 days:<br/>"
         "1. Duly filled claim form<br/>"
         "2. Original hospital bills and receipts<br/>"
         "3. Doctor's prescription and discharge summary<br/>"
         "4. Employee ID card copy<br/>"
         "Claims processed within 15 working days."),
    ])

    # --- 3. q1_fy25_financial_report.pdf ---
    _mkdoc("q1_fy25_financial_report.pdf", "TechCorp India — Q1 FY2025 Financial Report", [
        ("Reporting Period", "April 2025 - June 2025<br/>Prepared by: Finance<br/>Date: July 15, 2025"),
        ("Revenue Summary", ""),
        [[
            ["Segment", "Q1 FY2025 (₹ Cr)", "Q1 FY2024 (₹ Cr)", "YoY Growth"],
            ["SaaS Products", "45.2", "38.1", "18.6%"],
            ["Consulting Services", "22.8", "21.5", "6.0%"],
            ["License Revenue", "15.5", "14.2", "9.2%"],
            ["Maintenance", "8.3", "7.9", "5.1%"],
            ["Total Revenue", "91.8", "81.7", "12.4%"],
        ], [140, 100, 100, 80], []],
        ("Key Metrics",
         "Net Profit: ₹18.5 Crores (20.2% margin)<br/>"
         "EBITDA: ₹27.3 Crores (29.7% margin)<br/>"
         "Cash Reserves: ₹120 Crores<br/>"
         "Employee Count: 1,247<br/>"
         "New Customers Added: 38<br/>"
         "Customer Churn Rate: 2.3%"),
    ])

    # --- 4. hr_policy_handbook.pdf ---
    _mkdoc("hr_policy_handbook.pdf", "HR Policy Handbook 2025", [
        ("Leave Policy",
         "Annual Leave: 24 working days per year<br/>"
         "Sick Leave: 12 days per year (carry forward up to 30 days)<br/>"
         "Casual Leave: 12 days per year (lapse at year end)<br/>"
         "Maternity Leave: 26 weeks<br/>"
         "Paternity Leave: 5 working days<br/>"
         "Bereavement Leave: 3 days"),
        ("Work Hours",
         "Standard hours: 9:00 AM to 6:00 PM (Mon-Fri)<br/>"
         "Flexible timing available with manager approval<br/>"
         "Remote work: Up to 2 days per week<br/>"
         "Lunch break: 1 hour (12:30 PM - 2:30 PM)"),
        ("Code of Conduct",
         "Maintain confidentiality of company information<br/>"
         "Prohibition of insider trading<br/>"
         "Anti-harassment and non-discrimination policies<br/>"
         "Conflict of interest disclosure requirements<br/>"
         "Compliance with all applicable laws"),
    ])

    # --- 5. project_alpha_spec.pdf ---
    _mkdoc("project_alpha_spec.pdf", "Project Alpha — Technical Specification", [
        ("Project Info",
         "Project ID: PRJ-ALPHA-2025-001<br/>"
         "Product Owner: Priya Mehta<br/>"
         "Tech Lead: Hemant Sharma<br/>"
         "Architecture: Microservices on Kubernetes"),
        ("System Architecture",
         "Frontend: React 18 with TypeScript<br/>"
         "Backend: Python FastAPI (microservices)<br/>"
         "Database: PostgreSQL 15, Redis 7<br/>"
         "Message Queue: Apache Kafka 3.5<br/>"
         "Infrastructure: AWS EKS with Terraform<br/>"
         "CI/CD: GitLab CI with ArgoCD"),
        ("Budget Allocation", ""),
        [[
            ["Category", "Budget (₹ Lakhs)", "Spent (₹ Lakhs)", "Remaining"],
            ["Infrastructure", "45.0", "32.5", "12.5"],
            ["Licensing", "12.0", "10.2", "1.8"],
            ["Development", "78.0", "55.0", "23.0"],
            ["Testing & QA", "18.0", "12.0", "6.0"],
            ["Contingency", "10.0", "2.3", "7.7"],
            ["Total", "163.0", "112.0", "51.0"],
        ], [120, 100, 100, 100],
         [("BACKGROUND", (0, -1), (-1, -1), (0.91, 0.91, 0.91)),
          ("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold")]],
    ])

    # --- 6. board_meeting_minutes.pdf ---
    _mkdoc("board_meeting_minutes.pdf", "Board Meeting Minutes", [
        ("Meeting Info",
         "Date: June 20, 2025 | Time: 10:00 AM - 12:30 PM<br/>"
         "Venue: Board Room, 5th Floor, TechCorp Tower<br/>"
         "Chairperson: Anjali Verma (CEO)"),
        ("Attendees",
         "Anjali Verma (CEO) — Chairperson<br/>"
         "Rajesh Kumar (VP Engineering)<br/>"
         "Sunita Patel (CFO)<br/>"
         "Vikram Singh (CTO)<br/>"
         "Priya Mehta (Product Director)<br/>"
         "Amit Kapoor (HR Head)"),
        ("Agenda & Decisions",
         "1. Q1 FY2025 Financial Review — Approved<br/>"
         "2. Project Alpha Status Update — On track for Sept launch<br/>"
         "3. Employee Bonus Policy Revision — Deferred<br/>"
         "4. New Office Space Lease — Approved (Mumbai, BKC)<br/>"
         "5. Diversity Hiring Initiative — Approved (₹25 Lakh budget)"),
    ])

    # --- 7. hemant_performance_review.pdf ---
    _mkdoc("hemant_performance_review.pdf", "Employee Performance Review — H1 2025", [
        ("Employee Info",
         "Employee Name: Hemant Sharma<br/>"
         "Employee ID: EMP-2025-0042<br/>"
         "Review Period: Jan 2025 — June 2025<br/>"
         "Reviewer: Rajesh Kumar (VP Engineering)<br/>"
         "Rating: Exceeds Expectations (4.5/5)"),
        ("Key Achievements",
         "1. Led development of Project Alpha's core document processing engine<br/>"
         "2. Mentored 2 junior engineers through onboarding<br/>"
         "3. Reduced system latency by 40%<br/>"
         "4. Completed AWS Solutions Architect certification<br/>"
         "5. Received 'Star Performer' award for Q2 2025"),
        ("Compensation Review",
         "Recommended increment: 12%<br/>"
         "Proposed new monthly CTC: ₹1,40,000<br/>"
         "Performance bonus eligibility: Confirmed<br/>"
         "Next review: December 2025"),
    ])

    # --- 8. customer_satisfaction_report.pdf ---
    _mkdoc("customer_satisfaction_report.pdf", "Customer Satisfaction Report — Q2 FY2025", [
        ("Survey Period", "April — June 2025 | Prepared by: Customer Success Team"),
        ("Overall Metrics", ""),
        [[
            ["Metric", "Q2 FY2025", "Q1 FY2025", "Change"],
            ["NPS Score", "72", "68", "+4"],
            ["CSAT Score", "4.6/5", "4.4/5", "+0.2"],
            ["Resolution Time", "4.2 hrs", "5.1 hrs", "-17.6%"],
            ["First Response Time", "1.8 hrs", "2.3 hrs", "-21.7%"],
            ["Ticket Volume", "1,247", "1,382", "-9.8%"],
        ], [140, 90, 90, 80], []],
        ("Top Customer Complaints",
         "1. Mobile app crashes on Android 14 (23 reports)<br/>"
         "2. API rate limiting too restrictive (18 reports)<br/>"
         "3. Invoice generation delays (12 reports)<br/>"
         "4. Lack of offline mode (9 reports)"),
    ])

    # --- 9. it_asset_register.pdf ---
    _mkdoc("it_asset_register.pdf", "IT Asset Register — 2025", [
        ("Summary", "Department: IT Infrastructure | Last Updated: June 30, 2025 | Total Assets: 1,456"),
        ("Asset Summary by Category", ""),
        [[
            ["Category", "Count", "Total Value (₹ Lakhs)", "Avg Age (Years)"],
            ["Laptops", "420", "168.0", "1.8"],
            ["Desktops", "180", "54.0", "2.5"],
            ["Monitors", "380", "38.0", "2.0"],
            ["Servers", "45", "225.0", "3.1"],
            ["Network Equipment", "120", "60.0", "2.7"],
            ["Mobile Devices", "210", "42.0", "1.5"],
            ["Printers", "38", "9.5", "3.5"],
            ["Other Peripherals", "63", "12.6", "2.2"],
        ], [120, 70, 130, 100],
         [("BACKGROUND", (0, -1), (-1, -1), (0.91, 0.91, 0.91))]],
    ])

    # --- 10. vendor_agreement_cloudserve.pdf ---
    _mkdoc("vendor_agreement_cloudserve.pdf", "Vendor Service Agreement", [
        ("Agreement Info",
         "Agreement Number: VSA-2025-0042<br/>"
         "Date: May 1, 2025<br/>"
         "Vendor: CloudServe Technologies Pvt. Ltd.<br/>"
         "Service: Cloud Infrastructure & Managed Services"),
        ("Service Scope",
         "AWS Managed Services (24x7 monitoring)<br/>"
         "Database Administration (PostgreSQL, Redis)<br/>"
         "Security Audits (quarterly)<br/>"
         "Disaster Recovery Planning<br/>"
         "SLA: 99.95% uptime guarantee"),
        ("Pricing", ""),
        [[
            ["Service", "Monthly Fee (₹ Lakhs)", "Term"],
            ["AWS Management", "8.5", "12 months"],
            ["DBA Support", "3.2", "12 months"],
            ["Security Audit", "1.8", "Per audit"],
            ["DR Planning", "2.5", "One-time"],
        ], [140, 130, 100], []],
    ])

    print()
    print("=" * 60)
    print("  Generating questions.txt")
    print("=" * 60)

    qpath = Path("data/questions.txt")
    questions = [
        "What is Hemant Sharma's employee ID?",
        "Who is Hemant's reporting manager?",
        "What was Hemant's monthly salary?",
        "What is Clause 6.2(e)?",
        "Why was Hemant's bonus withheld?",
        "What is the NPS score for Q2 FY2025?",
        "What is the total value of laptops in the IT asset register?",
        "What is the monthly fee for AWS Management services from CloudServe?",
        "How many annual leave days are provided per the HR policy?",
        "What is the recommended increment percentage for Hemant?",
    ]
    qpath.write_text("\n".join(questions) + "\n", encoding="utf-8")
    print(f"  CREATED: {qpath.name} ({len(questions)} questions)")

    print()
    print("=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    pdfs = sorted(DATA_DIR.glob("*.pdf"))
    print(f"  Total PDFs: {len(pdfs)}")
    for p in pdfs:
        print(f"    {p.name} ({p.stat().st_size / 1024:.1f} KB)")
    print("  Done!")


if __name__ == "__main__":
    main()
