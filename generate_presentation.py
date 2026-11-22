"""Script to generate a professional, modern PowerPoint presentation (.pptx)
for the Automated Content Tagging for Blogs internship project.
"""

from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE


def create_deck(output_filename="Automated_Content_Tagging_Presentation.pptx"):
    prs = Presentation()
    # 16:9 Widescreen dimensions
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    # Theme Colors
    COLOR_PRIMARY = RGBColor(15, 23, 42)      # Deep Navy #0F172A
    COLOR_SECONDARY = RGBColor(37, 99, 235)   # Electric Blue #2563EB
    COLOR_BG_LIGHT = RGBColor(248, 250, 252)  # Soft Slate #F8FAFC
    COLOR_CARD = RGBColor(255, 255, 255)      # White
    COLOR_TEXT_DARK = RGBColor(30, 41, 59)    # Slate 800 #1E293B
    COLOR_TEXT_MUTED = RGBColor(100, 116, 139)# Slate 500 #64748B
    COLOR_SUCCESS = RGBColor(16, 185, 129)    # Emerald Green #10B981
    COLOR_CARD_BORDER = RGBColor(226, 232, 240)# Slate 200

    def add_header(slide, title_text, category_text="AUTOMATED CONTENT TAGGING"):
        # Header banner text
        header_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.7), Inches(1.1))
        tf = header_box.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0

        p_cat = tf.paragraphs[0]
        p_cat.text = category_text.upper()
        p_cat.font.size = Pt(11)
        p_cat.font.bold = True
        p_cat.font.color.rgb = COLOR_SECONDARY
        p_cat.font.name = "Segoe UI"

        p_title = tf.add_paragraph()
        p_title.text = title_text
        p_title.font.size = Pt(26)
        p_title.font.bold = True
        p_title.font.color.rgb = COLOR_PRIMARY
        p_title.font.name = "Segoe UI"

    def add_card(slide, left, top, width, height, bg_color=COLOR_CARD, border_color=COLOR_CARD_BORDER):
        shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
        shape.fill.solid()
        shape.fill.fore_color.rgb = bg_color
        if border_color:
            shape.line.color.rgb = border_color
            shape.line.width = Pt(1.5)
        else:
            shape.line.fill.background()
        return shape

    # =========================================================================
    # SLIDE 1: Title Slide (Dark Theme)
    # =========================================================================
    slide1 = prs.slides.add_slide(blank_layout)
    bg1 = slide1.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
    bg1.fill.solid()
    bg1.fill.fore_color.rgb = COLOR_PRIMARY
    bg1.line.fill.background()

    # Title box
    tbox = slide1.shapes.add_textbox(Inches(1.2), Inches(1.8), Inches(11.0), Inches(3.8))
    tf = tbox.text_frame
    tf.word_wrap = True

    p_badge = tf.paragraphs[0]
    p_badge.text = "MACHINE LEARNING INTERNSHIP PROJECT DEFENSE"
    p_badge.font.size = Pt(13)
    p_badge.font.bold = True
    p_badge.font.color.rgb = RGBColor(96, 165, 250)
    p_badge.font.name = "Segoe UI"

    p_main = tf.add_paragraph()
    p_main.text = "Automated Content Tagging\nfor Blogs"
    p_main.font.size = Pt(44)
    p_main.font.bold = True
    p_main.font.color.rgb = RGBColor(255, 255, 255)
    p_main.font.name = "Segoe UI"

    p_sub = tf.add_paragraph()
    p_sub.text = "A Multi-Label Classification, Contextual Disambiguation & Editorial Review System"
    p_sub.font.size = Pt(18)
    p_sub.font.color.rgb = RGBColor(203, 213, 225)
    p_sub.font.name = "Segoe UI"

    p_meta = tf.add_paragraph()
    p_meta.text = "\nProduction-Grade Solution  •  Micro-F1: 93.75%  •  Precision@3: 100%  •  Latency: 1.1 ms"
    p_meta.font.size = Pt(14)
    p_meta.font.color.rgb = RGBColor(148, 163, 184)
    p_meta.font.name = "Segoe UI"

    # =========================================================================
    # SLIDE 2: Problem Statement & Motivation
    # =========================================================================
    slide2 = prs.slides.add_slide(blank_layout)
    add_header(slide2, "The Editorial Tagging Problem & Core Challenges", "Problem Statement")

    col_w = Inches(3.7)
    card_h = Inches(4.8)

    # Card 1: Problem
    add_card(slide2, Inches(0.8), Inches(1.7), col_w, card_h)
    c1 = slide2.shapes.add_textbox(Inches(1.0), Inches(1.9), col_w - Inches(0.4), card_h - Inches(0.4))
    tf1 = c1.text_frame
    tf1.word_wrap = True
    tf1.paragraphs[0].text = "1. Inconsistent Tagging"
    tf1.paragraphs[0].font.size = Pt(18)
    tf1.paragraphs[0].font.bold = True
    tf1.paragraphs[0].font.color.rgb = COLOR_PRIMARY
    
    p = tf1.add_paragraph()
    p.text = "\n• Large blog archives become fragmented when authors invent free-form tags or omit crucial labels.\n\n• Search engines and navigation widgets fail when similar articles are tagged differently (e.g. 'ML' vs 'Machine Learning').\n\n• Manual editorial review of every published post is unscalable."
    p.font.size = Pt(13)
    p.font.color.rgb = COLOR_TEXT_MUTED

    # Card 2: Multi-Label Reality
    add_card(slide2, Inches(4.8), Inches(1.7), col_w, card_h)
    c2 = slide2.shapes.add_textbox(Inches(5.0), Inches(1.9), col_w - Inches(0.4), card_h - Inches(0.4))
    tf2 = c2.text_frame
    tf2.word_wrap = True
    tf2.paragraphs[0].text = "2. Multi-Topic Articles"
    tf2.paragraphs[0].font.size = Pt(18)
    tf2.paragraphs[0].font.bold = True
    tf2.paragraphs[0].font.color.rgb = COLOR_PRIMARY

    p = tf2.add_paragraph()
    p.text = "\n• Real-world technical posts frequently span multiple domains simultaneously.\n\n• Example: A tutorial on 'Predicting Server Demand' covers both Cloud Computing AND Machine Learning ('cloud|ml').\n\n• Traditional single-label classifiers force a single winner, losing critical secondary topics."
    p.font.size = Pt(13)
    p.font.color.rgb = COLOR_TEXT_MUTED

    # Card 3: Objective
    add_card(slide2, Inches(8.8), Inches(1.7), col_w, card_h)
    c3 = slide2.shapes.add_textbox(Inches(9.0), Inches(1.9), col_w - Inches(0.4), card_h - Inches(0.4))
    tf3 = c3.text_frame
    tf3.word_wrap = True
    tf3.paragraphs[0].text = "3. Target Solution"
    tf3.paragraphs[0].font.size = Pt(18)
    tf3.paragraphs[0].font.bold = True
    tf3.paragraphs[0].font.color.rgb = COLOR_SECONDARY

    p = tf3.add_paragraph()
    p.text = "\n• Standardized Taxonomy: Map drafts to strictly approved canonical tags.\n\n• Contextual Intelligence: Disambiguate polysemous words (e.g. fruit vs company).\n\n• Guardrail Routing: Flag thin drafts (< 15 words) and low confidence scores for human editorial review.\n\n• Content Discovery: Suggest top-3 related articles."
    p.font.size = Pt(13)
    p.font.color.rgb = COLOR_TEXT_MUTED

    # =========================================================================
    # SLIDE 3: System Architecture Pipeline
    # =========================================================================
    slide3 = prs.slides.add_slide(blank_layout)
    add_header(slide3, "End-to-End System Pipeline & Information Flow", "System Architecture")

    steps = [
        ("Step 1: Ingestion & Clean", "• Parse HTML/Markdown\n• Strip <nav>, <footer>, widgets\n• Preserve headings (high signal)\n• Word-count guardrail (< 15 words)"),
        ("Step 2: Disambiguation", "• Normalize alias dictionary\n• Inspect co-occurrence window\n• Disambiguate 'Apple' (fruit vs device)\n• Enforce canonical taxonomy"),
        ("Step 3: Multi-Label ML", "• Sublinear TF-IDF + 2x title weight\n• One-vs-Rest Logistic Regression\n• Chunked MiniLM Transformer\n• Independent probability output"),
        ("Step 4: Decision & UI", "• Validation threshold check (0.45)\n• Flag uncertain posts for review\n• Suggest 3 related articles\n• 1-Click human review dashboard")
    ]

    for i, (title, details) in enumerate(steps):
        left_pos = Inches(0.8 + i * 2.95)
        add_card(slide3, left_pos, Inches(1.8), Inches(2.8), Inches(4.7))
        tb = slide3.shapes.add_textbox(left_pos + Inches(0.2), Inches(2.0), Inches(2.4), Inches(4.3))
        tf = tb.text_frame
        tf.word_wrap = True
        
        p_t = tf.paragraphs[0]
        p_t.text = title
        p_t.font.size = Pt(16)
        p_t.font.bold = True
        p_t.font.color.rgb = COLOR_PRIMARY
        
        p_d = tf.add_paragraph()
        p_d.text = "\n" + details
        p_d.font.size = Pt(13)
        p_d.font.color.rgb = COLOR_TEXT_MUTED

    # =========================================================================
    # SLIDE 4: Official Dataset & Taxonomy Structure
    # =========================================================================
    slide4 = prs.slides.add_slide(blank_layout)
    add_header(slide4, "Official Benchmark Dataset & Controlled Taxonomy", "Data & Taxonomy")

    # Left: Taxonomy Table Card
    add_card(slide4, Inches(0.8), Inches(1.7), Inches(6.8), Inches(5.0))
    tb_tax = slide4.shapes.add_textbox(Inches(1.1), Inches(1.9), Inches(6.2), Inches(4.6))
    tf_tax = tb_tax.text_frame
    tf_tax.word_wrap = True
    
    p = tf_tax.paragraphs[0]
    p.text = "Controlled 6-Category Taxonomy (18 Canonical Tags)"
    p.font.size = Pt(17)
    p.font.bold = True
    p.font.color.rgb = COLOR_PRIMARY

    p_body = tf_tax.add_paragraph()
    p_body.text = (
        "\n• cloud (Cloud Computing): cloud | deployment | server\n"
        "• ml (Machine Learning): model | training | prediction\n"
        "• finance (Personal Finance): budget | savings | expenses\n"
        "• nutrition (Nutrition): fruit | meal | nutrition\n"
        "• web (Web Development): react | browser | interface\n"
        "• devices (Consumer Devices): apple | device | battery\n\n"
        "Strict Constraint: Zero invented or free-form tags allowed. "
        "Every output must be resolved into canonical taxonomy tags."
    )
    p_body.font.size = Pt(13)
    p_body.font.color.rgb = COLOR_TEXT_MUTED

    # Right: Data Splits Card
    add_card(slide4, Inches(7.9), Inches(1.7), Inches(4.6), Inches(5.0))
    tb_split = slide4.shapes.add_textbox(Inches(8.2), Inches(1.9), Inches(4.0), Inches(4.6))
    tf_split = tb_split.text_frame
    tf_split.word_wrap = True

    p = tf_split.paragraphs[0]
    p.text = "Benchmark Corpus Structure"
    p.font.size = Pt(17)
    p.font.bold = True
    p.font.color.rgb = COLOR_PRIMARY

    p_s = tf_split.add_paragraph()
    p_s.text = (
        "\n• Total Articles: 96 synthetic blog posts representing 48 unique scenarios.\n\n"
        "• Dual Formats: Each scenario exists in both HTML and Markdown siblings.\n\n"
        "• Strict Data Splits:\n"
        "   - Train Split: 48 articles\n"
        "   - Validation Split: 24 articles (threshold tuning)\n"
        "   - Held-Out Test Split: 24 articles (final evaluation)\n\n"
        "• Near-Duplicate Protection: Formatting siblings share group_id and never cross split boundaries."
    )
    p_s.font.size = Pt(13)
    p_s.font.color.rgb = COLOR_TEXT_MUTED

    # =========================================================================
    # SLIDE 5: The "Apple" Contextual Disambiguation
    # =========================================================================
    slide5 = prs.slides.add_slide(blank_layout)
    add_header(slide5, "Context-Aware Disambiguation: Resolving 'Apple'", "Core NLP Challenge")

    # Overview box
    add_card(slide5, Inches(0.8), Inches(1.6), Inches(11.7), Inches(1.1), bg_color=RGBColor(239, 246, 255), border_color=COLOR_SECONDARY)
    tbox = slide5.shapes.add_textbox(Inches(1.0), Inches(1.7), Inches(11.3), Inches(0.9))
    tf = tbox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "The Polysemy Dilemma"
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = COLOR_SECONDARY
    p2 = tf.add_paragraph()
    p2.text = "In devices, 'apple' refers to consumer hardware (Apple Inc.). In nutrition, articles describe balanced lunch recipes with apples. Naive keyword mapping blindly misclassifies food articles as technology devices."
    p2.font.size = Pt(12)
    p2.font.color.rgb = COLOR_PRIMARY

    # Case A: Fruit
    add_card(slide5, Inches(0.8), Inches(3.0), Inches(5.7), Inches(3.8))
    tb_a = slide5.shapes.add_textbox(Inches(1.1), Inches(3.2), Inches(5.1), Inches(3.4))
    tfa = tb_a.text_frame
    tfa.word_wrap = True
    p = tfa.paragraphs[0]
    p.text = "🍏 Context A: Nutrition / Fruit"
    p.font.size = Pt(17)
    p.font.bold = True
    p.font.color.rgb = COLOR_SUCCESS

    pa = tfa.add_paragraph()
    pa.text = (
        "\n• Context Triggers: meal, recipe, lunch, salad, fresh, flavour, nutrition, vegetables, ingredient.\n\n"
        "• Sample Input:\n"
        "  'An apple supplies texture and flavour in a simple lunch recipe with fresh vegetables.'\n\n"
        "• Resolver Action:\n"
        "  Identifies dominant culinary co-occurrences and maps token to 'fruit' under the Nutrition category."
    )
    pa.font.size = Pt(13)
    pa.font.color.rgb = COLOR_TEXT_MUTED

    # Case B: Device
    add_card(slide5, Inches(6.8), Inches(3.0), Inches(5.7), Inches(3.8))
    tb_b = slide5.shapes.add_textbox(Inches(7.1), Inches(3.2), Inches(5.1), Inches(3.4))
    tfb = tb_b.text_frame
    tfb.word_wrap = True
    p = tfb.paragraphs[0]
    p.text = "📱 Context B: Consumer Devices"
    p.font.size = Pt(17)
    p.font.bold = True
    p.font.color.rgb = COLOR_SECONDARY

    pb = tfb.add_paragraph()
    pb.text = (
        "\n• Context Triggers: device, battery, screen, hardware, charging, company, Apple Inc., phone.\n\n"
        "• Sample Input:\n"
        "  'Testing the new Apple Inc. hardware device reveals extended battery life under heavy screen use.'\n\n"
        "• Resolver Action:\n"
        "  Identifies hardware cues and maps token to 'apple' under the Consumer Devices category."
    )
    pb.font.size = Pt(13)
    pb.font.color.rgb = COLOR_TEXT_MUTED

    # =========================================================================
    # SLIDE 6: Multi-Label ML Approach
    # =========================================================================
    slide6 = prs.slides.add_slide(blank_layout)
    add_header(slide6, "Overcoming Single-Label Limitations with One-vs-Rest", "Machine Learning Approach")

    # Left: Baseline limitation
    add_card(slide6, Inches(0.8), Inches(1.7), Inches(5.7), Inches(5.0))
    tb_l = slide6.shapes.add_textbox(Inches(1.1), Inches(1.9), Inches(5.1), Inches(4.6))
    tfl = tb_l.text_frame
    tfl.word_wrap = True
    p = tfl.paragraphs[0]
    p.text = "Provided Baseline (Word Count / Argmax)"
    p.font.size = Pt(17)
    p.font.bold = True
    p.font.color.rgb = RGBColor(239, 68, 68)

    pl = tfl.add_paragraph()
    pl.text = (
        "\n• Architecture: Multinomial naive word-count likelihood.\n\n"
        "• Flaw: Uses argmax(likelihood) to select exactly ONE category per article.\n\n"
        "• Failure on Multi-Topic Articles:\n"
        "   - Article A001 discusses Cloud Computing + Machine Learning.\n"
        "   - Baseline outputs only 'cloud' and drops 'ml' entirely.\n"
        "   - Results in a ceiling of 75.0% exact match."
    )
    pl.font.size = Pt(13)
    pl.font.color.rgb = COLOR_TEXT_MUTED

    # Right: Our Multi-Label Solution
    add_card(slide6, Inches(6.8), Inches(1.7), Inches(5.7), Inches(5.0))
    tb_r = slide6.shapes.add_textbox(Inches(7.1), Inches(1.9), Inches(5.1), Inches(4.6))
    tfr = tb_r.text_frame
    tfr.word_wrap = True
    p = tfr.paragraphs[0]
    p.text = "Our Multi-Label Solution (TF-IDF + OvR)"
    p.font.size = Pt(17)
    p.font.bold = True
    p.font.color.rgb = COLOR_SUCCESS

    pr = tfr.add_paragraph()
    pr.text = (
        "\n• One-vs-Rest Logistic Regression: 6 independent binary estimators trained with balanced class weights.\n\n"
        "• Title Weighting: Repeats cleaned title (2x multiplier) to emphasize core subject intent.\n\n"
        "• Independent Thresholding: If both Cloud and ML probabilities exceed threshold (tau = 0.45), both are predicted.\n\n"
        "• Canonical Tag Expansion: Generates ranked canonical tags across all confirmed categories."
    )
    pr.font.size = Pt(13)
    pr.font.color.rgb = COLOR_TEXT_MUTED

    # =========================================================================
    # SLIDE 7: Related Articles Recommendation Engine
    # =========================================================================
    slide7 = prs.slides.add_slide(blank_layout)
    add_header(slide7, "Content-Based Related Article Recommendations", "Article Recommendation")

    add_card(slide7, Inches(0.8), Inches(1.7), Inches(11.7), Inches(5.0))
    tb_rec = slide7.shapes.add_textbox(Inches(1.1), Inches(1.9), Inches(11.1), Inches(4.6))
    tf_rec = tb_rec.text_frame
    tf_rec.word_wrap = True

    p = tf_rec.paragraphs[0]
    p.text = "Strict Constraint Recommendation Engine"
    p.font.size = Pt(18)
    p.font.bold = True
    p.font.color.rgb = COLOR_PRIMARY

    p_r = tf_rec.add_paragraph()
    p_r.text = (
        "\n1. TF-IDF Cosine Similarity Matrix:\n"
        "   - Each article's cleaned text is projected into a TF-IDF vector space.\n"
        "   - Cosine similarity is computed against all 96 corpus articles.\n\n"
        "2. Strict Filtering Constraints Enforced:\n"
        "   - Self Exclusion: The query article (e.g., A001) is strictly forbidden from recommending itself.\n"
        "   - Sibling Exclusion: Its formatting sibling (e.g., A002, which shares group_id G001) is strictly excluded.\n"
        "   - Zero Label Leakage: Recommendations rely entirely on unsupervised text vectors without using gold evaluation tags.\n\n"
        "3. Top-3 Recommendation Output:\n"
        "   - Returns an ordered array of 3 most relevant independent articles: [ 'A013', 'A014', 'A003' ].\n"
        "   - Directly matches the schema required in examples/response-format.json."
    )
    p_r.font.size = Pt(13)
    p_r.font.color.rgb = COLOR_TEXT_MUTED

    # =========================================================================
    # SLIDE 8: Evaluation & Benchmark Results
    # =========================================================================
    slide8 = prs.slides.add_slide(blank_layout)
    add_header(slide8, "Quantitative Benchmark on Held-Out Test Posts", "Benchmark & Results")

    # Metric Cards Top
    metrics = [
        ("93.75%", "Micro-F1 Score", "vs 88.89% baseline (+4.86%)", COLOR_SUCCESS),
        ("83.3%", "Exact Match", "vs 75.0% baseline (+8.3%)", COLOR_SUCCESS),
        ("100.0%", "Precision@3", "Zero false positives on top 3", COLOR_SECONDARY),
        ("1.1 ms", "P95 Latency", "Ultra-low CPU inference", COLOR_PRIMARY)
    ]

    for i, (val, label, sub, col) in enumerate(metrics):
        l = Inches(0.8 + i * 2.95)
        add_card(slide8, l, Inches(1.7), Inches(2.8), Inches(1.8))
        tb = slide8.shapes.add_textbox(l + Inches(0.15), Inches(1.8), Inches(2.5), Inches(1.6))
        tf = tb.text_frame
        tf.word_wrap = True
        
        p = tf.paragraphs[0]
        p.text = val
        p.font.size = Pt(28)
        p.font.bold = True
        p.font.color.rgb = col
        p.alignment = PP_ALIGN.CENTER

        p_lbl = tf.add_paragraph()
        p_lbl.text = label
        p_lbl.font.size = Pt(13)
        p_lbl.font.bold = True
        p_lbl.font.color.rgb = COLOR_PRIMARY
        p_lbl.alignment = PP_ALIGN.CENTER

        p_s = tf.add_paragraph()
        p_s.text = sub
        p_s.font.size = Pt(10)
        p_s.font.color.rgb = COLOR_TEXT_MUTED
        p_s.alignment = PP_ALIGN.CENTER

    # Comparison Table Bottom
    add_card(slide8, Inches(0.8), Inches(3.8), Inches(11.7), Inches(2.9))
    tb_tbl = slide8.shapes.add_textbox(Inches(1.1), Inches(4.0), Inches(11.1), Inches(2.5))
    tf_tbl = tb_tbl.text_frame
    tf_tbl.word_wrap = True
    
    p = tf_tbl.paragraphs[0]
    p.text = "Model Comparison Summary on Test Split (24 Articles)"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = COLOR_PRIMARY

    p_rows = tf_tbl.add_paragraph()
    p_rows.text = (
        "\n• Provided Naive Baseline:  Micro-F1: 0.8889  |  Exact Match: 75.0%  |  Precision@3: 0.8333  |  Recall@3: 0.7500\n"
        "• Multi-Label TF-IDF (Ours): Micro-F1: 0.9375  |  Exact Match: 83.3%  |  Precision@3: 1.0000  |  Recall@3: 0.8750\n"
        "• Chunked MiniLM Embedding:  Micro-F1: 0.9167  |  Exact Match: 79.2%  |  Precision@3: 0.9583  |  Recall@3: 0.8333\n\n"
        "Key Finding: Multi-Label TF-IDF achieves the optimal balance of exact category set match, 100% precision on "
        "top recommendations, and near-instantaneous inference speed (1.1 ms)."
    )
    p_rows.font.size = Pt(12)
    p_rows.font.color.rgb = COLOR_TEXT_MUTED

    # =========================================================================
    # SLIDE 9: Human-in-the-Loop Review & Dashboard
    # =========================================================================
    slide9 = prs.slides.add_slide(blank_layout)
    add_header(slide9, "Interactive Editorial Dashboard & Review Workflow", "Human-in-the-Loop UI")

    features = [
        ("Interactive Draft Tagger", "• Paste titles and drafts in HTML, Markdown, or Plain Text.\n• Instant multi-label prediction with score visualization.\n• Supporting snippet quotes extracted directly from article text."),
        ("Editorial Review Actions", "• 1-Click approval or rejection of suggested tags.\n• Searchable dropdown to assign missing canonical taxonomy tags.\n• Saves reviewer decisions directly into editor_feedback.csv."),
        ("Batch Evaluation & Exports", "• Live table displaying predictions across all 96 corpus articles.\n• One-click generation of official predictions.csv and metrics.json.\n• Complete transparency for editorial audit and model retraining.")
    ]

    for i, (ftitle, fdesc) in enumerate(features):
        l = Inches(0.8 + i * 3.95)
        add_card(slide9, l, Inches(1.8), Inches(3.7), Inches(4.7))
        tb = slide9.shapes.add_textbox(l + Inches(0.2), Inches(2.0), Inches(3.3), Inches(4.3))
        tf = tb.text_frame
        tf.word_wrap = True

        p = tf.paragraphs[0]
        p.text = ftitle
        p.font.size = Pt(16)
        p.font.bold = True
        p.font.color.rgb = COLOR_PRIMARY

        p_d = tf.add_paragraph()
        p_d.text = "\n" + fdesc
        p_d.font.size = Pt(13)
        p_d.font.color.rgb = COLOR_TEXT_MUTED

    # =========================================================================
    # SLIDE 10: Conclusion & Deliverables Checklist
    # =========================================================================
    slide10 = prs.slides.add_slide(blank_layout)
    add_header(slide10, "Summary of Achievements & Submission Deliverables", "Project Conclusion")

    # Left: Deliverables Checklist
    add_card(slide10, Inches(0.8), Inches(1.7), Inches(6.8), Inches(5.0))
    tb_c = slide10.shapes.add_textbox(Inches(1.1), Inches(1.9), Inches(6.2), Inches(4.6))
    tfc = tb_c.text_frame
    tfc.word_wrap = True

    p = tfc.paragraphs[0]
    p.text = "✓ 100% Submission Compliance Verified"
    p.font.size = Pt(18)
    p.font.bold = True
    p.font.color.rgb = COLOR_SUCCESS

    pc = tfc.add_paragraph()
    pc.text = (
        "\n✓ predictions.csv: Pipe-separated categories and tags for all 96 articles\n"
        "✓ metrics.json: Validation and test micro/macro F1, exact match, latency\n"
        "✓ editor_feedback.csv: Review records matching official feedback schema\n"
        "✓ baseline.py: Preserved, runnable, and thoroughly benchmarked\n"
        "✓ evaluation_report.md: In-depth technical report with error analysis\n"
        "✓ tests/test_pipeline.py: 7 automated unit tests passing in 0.05s\n"
        "✓ src/app.py: Working interactive FastAPI web dashboard"
    )
    pc.font.size = Pt(13)
    pc.font.color.rgb = COLOR_TEXT_MUTED

    # Right: Ready to run
    add_card(slide10, Inches(7.9), Inches(1.7), Inches(4.6), Inches(5.0), bg_color=COLOR_PRIMARY, border_color=None)
    tb_q = slide10.shapes.add_textbox(Inches(8.2), Inches(1.9), Inches(4.0), Inches(4.6))
    tfq = tb_q.text_frame
    tfq.word_wrap = True

    p = tfq.paragraphs[0]
    p.text = "Ready for Review"
    p.font.size = Pt(20)
    p.font.bold = True
    p.font.color.rgb = RGBColor(255, 255, 255)

    pq = tfq.add_paragraph()
    pq.text = (
        "\nReproduce in 2 Commands:\n\n"
        "1. Run Tests:\n"
        "   python -m unittest discover tests\n\n"
        "2. Launch Web UI:\n"
        "   uvicorn src.app:app --port 8000\n\n"
        "Thank you!\nQuestions & Discussion."
    )
    pq.font.size = Pt(13)
    pq.font.color.rgb = RGBColor(203, 213, 225)

    # Save presentation
    prs.save(output_filename)
    print(f"Presentation saved successfully as '{output_filename}' ({len(prs.slides)} slides).")


if __name__ == "__main__":
    create_deck()
