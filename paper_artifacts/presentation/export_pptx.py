"""
Generate a professional, publication-grade PowerPoint (16:9) presentation for
Cognitive State Compilation (CSC).
"""
import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE

OUTPUT_PATH = r"c:\Projects\cognitive_os_1\presentation\CSC_Paper_Presentation.pptx"
FIGS_DIR = r"c:\Projects\cognitive_os_1\presentation\figs"

# Theme Colors (Deep Slate, Vibrant Indigo, Cyan, Emerald, Muted Text)
BG_COLOR = RGBColor(15, 23, 42)        # Slate 900 #0f172a
CARD_BG = RGBColor(30, 41, 59)        # Slate 800 #1e293b
CARD_BORDER = RGBColor(51, 65, 85)    # Slate 700 #334155
PRIMARY = RGBColor(99, 102, 241)      # Indigo 500 #6366f1
ACCENT_CYAN = RGBColor(56, 189, 248)  # Cyan 400 #38bdf8
ACCENT_GREEN = RGBColor(52, 211, 153) # Emerald 400 #34d399
TEXT_WHITE = RGBColor(248, 250, 252)  # Slate 50 #f8fafc
TEXT_MUTED = RGBColor(148, 163, 184)  # Slate 400 #94a3b8
TEXT_DARK = RGBColor(15, 23, 42)

def set_slide_background(slide, prs):
    bg_shape = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height
    )
    bg_shape.fill.solid()
    bg_shape.fill.fore_color.rgb = BG_COLOR
    bg_shape.line.fill.background()
    return bg_shape

def add_header(slide, title_text, category="NEURIPS 2026 SUBMISSION • RESEARCH PRESENTATION"):
    # Category tag
    cat_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.5), Inches(0.35))
    tf_cat = cat_box.text_frame
    tf_cat.word_wrap = True
    p_cat = tf_cat.paragraphs[0]
    p_cat.text = category.upper()
    p_cat.font.size = Pt(11)
    p_cat.font.bold = True
    p_cat.font.color.rgb = ACCENT_CYAN
    p_cat.font.name = "Segoe UI"
    
    # Title
    t_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.7), Inches(11.5), Inches(0.7))
    tf_t = t_box.text_frame
    tf_t.word_wrap = True
    p_t = tf_t.paragraphs[0]
    p_t.text = title_text
    p_t.font.size = Pt(24)
    p_t.font.bold = True
    p_t.font.color.rgb = TEXT_WHITE
    p_t.font.name = "Segoe UI"

def add_card(slide, left, top, width, height, bg=CARD_BG, border=CARD_BORDER):
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    shape.fill.solid()
    shape.fill.fore_color.rgb = bg
    shape.line.color.rgb = border
    shape.line.width = Pt(1)
    return shape

def add_bullet_point(text_frame, bold_prefix, text_content, font_size=13, is_first=False):
    p = text_frame.paragraphs[0] if is_first else text_frame.add_paragraph()
    p.space_after = Pt(8)
    p.font.size = Pt(font_size)
    p.font.name = "Segoe UI"
    
    run1 = p.add_run()
    run1.text = bold_prefix + (" " if bold_prefix else "")
    run1.font.bold = True
    run1.font.color.rgb = TEXT_WHITE
    
    run2 = p.add_run()
    run2.text = text_content
    run2.font.bold = False
    run2.font.color.rgb = TEXT_MUTED

def set_notes(slide, notes_text):
    notes_slide = slide.notes_slide
    text_frame = notes_slide.notes_text_frame
    text_frame.text = notes_text

def build_presentation():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]
    
    # ==========================================
    # SLIDE 1: Title Slide
    # ==========================================
    s1 = prs.slides.add_slide(blank_layout)
    set_slide_background(s1, prs)
    
    card1 = add_card(s1, Inches(1.5), Inches(1.5), Inches(10.333), Inches(4.5), bg=CARD_BG, border=PRIMARY)
    tb1 = s1.shapes.add_textbox(Inches(1.8), Inches(1.8), Inches(9.7), Inches(3.9))
    tf1 = tb1.text_frame
    tf1.word_wrap = True
    
    p = tf1.paragraphs[0]
    p.text = "RESEARCH PAPER PRESENTATION"
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = ACCENT_CYAN
    p.font.name = "Segoe UI"
    p.space_after = Pt(14)
    
    p2 = tf1.add_paragraph()
    p2.text = "Cognitive State Compilation"
    p2.font.size = Pt(36)
    p2.font.bold = True
    p2.font.color.rgb = TEXT_WHITE
    p2.font.name = "Segoe UI"
    
    p3 = tf1.add_paragraph()
    p3.text = "Resource-Bounded Construction of Reasoning States for Long-Horizon LLM Agents"
    p3.font.size = Pt(20)
    p3.font.color.rgb = ACCENT_GREEN
    p3.font.name = "Segoe UI"
    p3.space_after = Pt(24)
    
    p4 = tf1.add_paragraph()
    p4.text = "Formalizing Reasoning Context Assembly as an Inference-Time 0/1 Knapsack Optimization"
    p4.font.size = Pt(14)
    p4.font.color.rgb = TEXT_MUTED
    p4.font.name = "Segoe UI"
    p4.space_after = Pt(16)
    
    p5 = tf1.add_paragraph()
    p5.text = "Track: Agent Architectures & Efficient Inference | NeurIPS 2026"
    p5.font.size = Pt(12)
    p5.font.bold = True
    p5.font.color.rgb = PRIMARY
    p5.font.name = "Segoe UI"
    
    set_notes(s1, 
        "Welcome everyone. Today I'm presenting Cognitive State Compilation, or CSC. "
        "This work addresses the critical bottleneck in long-horizon LLM agents: the Token Budget Crisis. "
        "Rather than treating memory as an unstructured buffer or relying on single-source RAG, "
        "we formalize the construction of the transient reasoning state as a constrained 0/1 knapsack "
        "over heterogeneous pools of knowledge, experience, and computation."
    )
    
    # ==========================================
    # SLIDE 2: Executive Summary & Highlights
    # ==========================================
    s2 = prs.slides.add_slide(blank_layout)
    set_slide_background(s2, prs)
    add_header(s2, "Executive Summary: Key Findings & Paradigm Shift")
    
    # 3 Stat Cards
    stat_data = [
        ("100% Coverage", "@ B = 500 Tokens", "CSC reaches 100% state coverage on HotpotQA vs 83.3% RAG and 60% MEM1.", ACCENT_CYAN),
        ("88.9% Live F1", "Downstream LLM Impact", "Evaluated on Groq Qwen3.6-27B vs 66.7% for the strongest baseline.", ACCENT_GREEN),
        ("100% Horizon Retention", "Zero RL Overhead", "Maintains 100% retention at horizon N=16 within 512 tokens while MEM1 drops to 82%.", PRIMARY)
    ]
    for i, (metric, sub, desc, col) in enumerate(stat_data):
        left = Inches(0.8 + i * 3.95)
        add_card(s2, left, Inches(1.5), Inches(3.75), Inches(2.2))
        tb = s2.shapes.add_textbox(left + Inches(0.2), Inches(1.65), Inches(3.35), Inches(1.9))
        tf = tb.text_frame
        tf.word_wrap = True
        
        p = tf.paragraphs[0]
        p.text = metric
        p.font.size = Pt(24)
        p.font.bold = True
        p.font.color.rgb = col
        p.font.name = "Segoe UI"
        
        p2 = tf.add_paragraph()
        p2.text = sub
        p2.font.size = Pt(12)
        p2.font.bold = True
        p2.font.color.rgb = TEXT_WHITE
        p2.space_after = Pt(8)
        
        p3 = tf.add_paragraph()
        p3.text = desc
        p3.font.size = Pt(11)
        p3.font.color.rgb = TEXT_MUTED
        
    # Lower takeaway card
    add_card(s2, Inches(0.8), Inches(3.9), Inches(11.733), Inches(3.0))
    tb = s2.shapes.add_textbox(Inches(1.0), Inches(4.0), Inches(11.333), Inches(2.7))
    tf = tb.text_frame
    tf.word_wrap = True
    
    add_bullet_point(tf, "The Core Problem:", "Long-horizon agents accumulate unbounded history, but inference token budgets remain strictly capped.", 13, True)
    add_bullet_point(tf, "The Compilation Formulation:", "Instead of independent retrieval or recurrent compression, formulate state assembly as an explicit 0/1 knapsack over Knowledge, Experience, and Computation.", 13)
    add_bullet_point(tf, "Trained Scorer & Greedy Selector:", "A 9-feature Gradient Boosted Tree utility scorer feeds a greedy density selector running in O(N log N) time with 0.0% optimality gap.", 13)
    add_bullet_point(tf, "Scientific Transparency:", "We openly disclose limitations: in-distribution training label alignment and curated WebShop sample size.", 13)
    
    set_notes(s2, 
        "To summarize our top-line results: CSC achieves 100% state coverage at just 500 tokens, "
        "translating to an 88.9% F1 on live LLM evaluation with Groq Qwen3.6-27B. "
        "Crucially, on long-horizon multi-turn benchmarks, CSC maintains 100% state retention up to horizon N=16 "
        "within a strict 512-token envelope, outperforming MEM1 without any reinforcement learning training."
    )

    # ==========================================
    # SLIDE 3: The Token Budget Crisis
    # ==========================================
    s3 = prs.slides.add_slide(blank_layout)
    set_slide_background(s3, prs)
    add_header(s3, "The Token Budget Crisis in Long-Horizon LLM Agents")
    
    # Left Card: The Dilemma
    add_card(s3, Inches(0.8), Inches(1.5), Inches(5.7), Inches(5.4))
    tb_l = s3.shapes.add_textbox(Inches(1.0), Inches(1.7), Inches(5.3), Inches(5.0))
    tf_l = tb_l.text_frame
    tf_l.word_wrap = True
    
    add_bullet_point(tf_l, "Unbounded Accumulation:", "As agents act over long horizons, candidate history grows monotonically: external documents, execution traces, scratchpads, and memories.", 13, True)
    add_bullet_point(tf_l, "Fixed Inference Budgets:", "LLM context windows and inference latency/cost force a strict token ceiling B per reasoning step.", 13)
    add_bullet_point(tf_l, "The 'Lost in the Middle' Trap:", "Stuffing full context degrades reasoning, introduces distractor noise, and skyrockets quadratic attention costs.", 13)
    add_bullet_point(tf_l, "Homogeneous Fallacy:", "Real agent contexts are not just documents—they are a heterogeneous blend of static facts, past actions, and dynamic tool computations.", 13)
    
    # Right Card: The 3 Resource Modalities
    add_card(s3, Inches(6.8), Inches(1.5), Inches(5.7), Inches(5.4))
    tb_r = s3.shapes.add_textbox(Inches(7.0), Inches(1.7), Inches(5.3), Inches(5.0))
    tf_r = tb_r.text_frame
    tf_r.word_wrap = True
    
    p = tf_r.paragraphs[0]
    p.text = "The Heterogeneous Triad: R = R_know ∪ R_exp ∪ R_comp"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = ACCENT_CYAN
    p.space_after = Pt(12)
    
    add_bullet_point(tf_r, "1. Knowledge (R_know):", "Retrieved facts, Wikipedia passages, domain corpus docs. High density of ground truth, but verbose.", 12)
    add_bullet_point(tf_r, "2. Experience (R_exp):", "Few-shot trajectories, successful workflows, episodic reflection notes. Guides reasoning structure.", 12)
    add_bullet_point(tf_r, "3. Computation (R_comp):", "Intermediate scratchpad calculations, tool outputs, entity extractions, active state filters. Highly compact.", 12)
    add_bullet_point(tf_r, "The Core Question:", "Given budget B, how do we choose the optimal subset S ⊆ R to construct the transient reasoning prompt?", 12)

    set_notes(s3, 
        "Why does long-horizon agency break down? Agents accumulate past thoughts, search results, and tool outputs. "
        "Putting everything into the prompt causes latency explosion and reasoning degradation. "
        "Furthermore, agents don't just consume text—they balance three fundamentally different resource pools: "
        "Knowledge, Experience, and Computation. We need a compiler that selects across all three."
    )

    # ==========================================
    # SLIDE 4: Architectural Taxonomy
    # ==========================================
    s4 = prs.slides.add_slide(blank_layout)
    set_slide_background(s4, prs)
    add_header(s4, "Positioning Matrix: Memory Systems vs CSC")
    
    # Comparison Table
    table_shape = s4.shapes.add_table(8, 7, Inches(0.8), Inches(1.6), Inches(11.733), Inches(5.2))
    table = table_shape.table
    table.columns[0].width = Inches(2.2)
    table.columns[1].width = Inches(1.3)
    table.columns[2].width = Inches(1.3)
    table.columns[3].width = Inches(1.5)
    table.columns[4].width = Inches(1.4)
    table.columns[5].width = Inches(1.4)
    table.columns[6].width = Inches(2.633)
    
    headers = ["System", "Store", "Retrieve", "Compute Reuse", "Budget Aware", "Cross-Resource", "Primary Limitation vs CSC"]
    for c_idx, h in enumerate(headers):
        cell = table.cell(0, c_idx)
        cell.fill.solid()
        cell.fill.fore_color.rgb = CARD_BG
        p = cell.text_frame.paragraphs[0]
        p.text = h
        p.font.bold = True
        p.font.size = Pt(11)
        p.font.color.rgb = ACCENT_CYAN
        
    rows = [
        ("MemGPT (Packer et al.)", "Yes", "Yes", "No", "No", "No", "LLM-paged OS memory; no joint knapsack"),
        ("A-MEM (Xu et al.)", "Yes", "Yes", "No", "Partial", "No", "Linked notes; single homogeneous memory type"),
        ("Memory-R1 (Yan et al.)", "Yes", "Yes", "No", "No", "No", "RL editing policy; no budget-aware compiler"),
        ("AgeMem (Yu et al.)", "Yes", "Yes", "No", "Partial", "Partial", "Tiers LT/ST memory; lacks knapsack optimization"),
        ("MemOS (Li et al.)", "Yes", "Yes", "Yes", "No", "Partial", "OS memory lifecycle; lacks learned utility/budget"),
        ("MEM1 (Zhou et al.)", "Yes", "N/A", "Partial", "Yes", "Yes", "Black-box recurrent state via costly RL"),
        ("CSC (Ours)", "Yes", "Yes", "Yes", "Yes", "Yes", "Inference-time 0/1 knapsack + learned utility")
    ]
    
    for r_idx, row_vals in enumerate(rows):
        is_csc = (r_idx == len(rows) - 1)
        for c_idx, val in enumerate(row_vals):
            cell = table.cell(r_idx + 1, c_idx)
            cell.fill.solid()
            cell.fill.fore_color.rgb = RGBColor(30, 48, 80) if is_csc else (RGBColor(24, 32, 47) if r_idx % 2 == 0 else CARD_BG)
            p = cell.text_frame.paragraphs[0]
            p.text = val
            p.font.size = Pt(10)
            p.font.bold = is_csc
            p.font.color.rgb = ACCENT_GREEN if is_csc else (TEXT_WHITE if c_idx == 0 else TEXT_MUTED)

    set_notes(s4, 
        "This matrix compares CSC directly against prior long-horizon systems like MemGPT, A-MEM, and MEM1. "
        "Prior works optimize memory in silos: how to store notes, how to do top-k retrieval, or how to train RL policies. "
        "CSC is the first system to formalize state assembly as an explicit, cross-resource knapsack problem "
        "that jointly considers knowledge, experience, and computation under a hard token budget."
    )

    # ==========================================
    # SLIDE 5: Knapsack Formulation
    # ==========================================
    s5 = prs.slides.add_slide(blank_layout)
    set_slide_background(s5, prs)
    add_header(s5, "Mathematical Formulation: The 0/1 Knapsack Objective")
    
    # Left Card: Formulation
    add_card(s5, Inches(0.8), Inches(1.5), Inches(5.7), Inches(5.4))
    tb_l = s5.shapes.add_textbox(Inches(1.0), Inches(1.7), Inches(5.3), Inches(5.0))
    tf_l = tb_l.text_frame
    tf_l.word_wrap = True
    
    p = tf_l.paragraphs[0]
    p.text = "Constrained Optimization Problem"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = ACCENT_CYAN
    p.space_after = Pt(14)
    
    add_bullet_point(tf_l, "Objective:", "max_{S ⊆ R} ∑_{r ∈ S} u(r | τ, C)", 14)
    add_bullet_point(tf_l, "Subject to:", "∑_{r ∈ S} cost(r) ≤ B", 14)
    add_bullet_point(tf_l, "Optional Type Caps:", "counts[t] ≤ c_t  ∀ t ∈ {know, exp, comp}", 12)
    add_bullet_point(tf_l, "Item Payloads:", "Each candidate r ∈ R carries text payload, resource type, token cost(r), and optional hint h(r).", 12)
    add_bullet_point(tf_l, "Cost Estimation:", "Computed via standard subword tokenizer (tiktoken / BPE) with whitespace fallback.", 12)
    
    # Right Card: The Utility-Density Metric
    add_card(s5, Inches(6.8), Inches(1.5), Inches(5.7), Inches(5.4))
    tb_r = s5.shapes.add_textbox(Inches(7.0), Inches(1.7), Inches(5.3), Inches(5.0))
    tf_r = tb_r.text_frame
    tf_r.word_wrap = True
    
    p = tf_r.paragraphs[0]
    p.text = "Dantzig's Value-to-Cost Density Principle"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = ACCENT_GREEN
    p.space_after = Pt(14)
    
    add_bullet_point(tf_r, "Utility Density ρ(r):", "ρ(r) = u(r | τ) / max(1, cost(r))", 14)
    add_bullet_point(tf_r, "Why Density Beats Pure Utility:", "A 500-token document with utility 0.9 yields density 0.0018. A 50-token scratchpad with utility 0.7 yields density 0.0140 (7.7x higher ROI!).", 12)
    add_bullet_point(tf_r, "Greedy Packing:", "Sort candidates descending by ρ(r) and pack greedily until budget B is reached.", 12)
    add_bullet_point(tf_r, "Guaranteed Time Complexity:", "O(N log N) time complexity vs O(N · B) for exact dynamic programming.", 12)

    set_notes(s5, 
        "Here is the mathematical core. We pose state construction as a 0/1 knapsack. "
        "Given task tau and budget B, we select subset S that maximizes utility sum under the budget ceiling. "
        "Crucially, we use Dantzig's utility density: utility divided by token cost. "
        "A compact tool result that costs 40 tokens can offer far higher return-on-tokens than a rambling 500-token document."
    )

    # ==========================================
    # SLIDE 6: System Pipeline Architecture
    # ==========================================
    s6 = prs.slides.add_slide(blank_layout)
    set_slide_background(s6, prs)
    add_header(s6, "Modular Architecture: Scoring, Selection & Composition")
    
    steps = [
        ("1. Candidate Pool", "R_know: Retrieved docs\nR_exp: Past trajectories\nR_comp: Tool/scratchpads", PRIMARY),
        ("2. Feature Extractor", "9-Dimensional vector:\nTF-IDF (1/2-gram), Overlap,\nType bits, Hint, Cost density", ACCENT_CYAN),
        ("3. Production Scorer", "Gradient Boosted Trees\nP(useful | τ, r)\n(98.6% Acc, 94.2% F1)\nFallback: Dense Heuristic", ACCENT_GREEN),
        ("4. Knapsack Selector", "Greedy Density Sorting\nρ(r) = u(r) / cost(r)\nEnforces Budget B\nEnforces Type Caps c_t", PRIMARY),
        ("5. State Composer", "Tag & serialize items\nPrefix task query τ\nCompiled state S*\nFed to Target LLM", ACCENT_CYAN)
    ]
    
    for i, (stitle, sdesc, scol) in enumerate(steps):
        left = Inches(0.8 + i * 2.37)
        add_card(s6, left, Inches(1.6), Inches(2.2), Inches(5.2))
        tb = s6.shapes.add_textbox(left + Inches(0.1), Inches(1.8), Inches(2.0), Inches(4.8))
        tf = tb.text_frame
        tf.word_wrap = True
        
        p = tf.paragraphs[0]
        p.text = stitle
        p.font.size = Pt(13)
        p.font.bold = True
        p.font.color.rgb = scol
        p.font.name = "Segoe UI"
        p.space_after = Pt(12)
        
        p2 = tf.add_paragraph()
        p2.text = sdesc
        p2.font.size = Pt(11)
        p2.font.color.rgb = TEXT_WHITE
        p2.font.name = "Segoe UI"
        
    set_notes(s6, 
        "This slide depicts the end-to-end modular pipeline. "
        "It consists of five decoupled stages: Candidate aggregation, 9-D Feature Extraction, "
        "GBT Scoring, Greedy Knapsack Selection, and State Composition. "
        "Because each module has a clean API, the scorer can be swapped without touching the selector, "
        "and the greedy selector can be swapped for exact dynamic programming if needed."
    )

    # ==========================================
    # SLIDE 7: 9-D Features & Production Scorer
    # ==========================================
    s7 = prs.slides.add_slide(blank_layout)
    set_slide_background(s7, prs)
    add_header(s7, "Utility Scorer: 9-D Feature Representation")
    
    # Left Card: Table of Features
    add_card(s7, Inches(0.8), Inches(1.5), Inches(6.8), Inches(5.4))
    table_shape = s7.shapes.add_table(10, 3, Inches(1.0), Inches(1.7), Inches(6.4), Inches(5.0))
    table = table_shape.table
    table.columns[0].width = Inches(0.8)
    table.columns[1].width = Inches(2.8)
    table.columns[2].width = Inches(2.8)
    
    f_headers = ["Dim", "Feature Name", "Semantic Role"]
    for c_idx, h in enumerate(f_headers):
        cell = table.cell(0, c_idx)
        cell.fill.solid()
        cell.fill.fore_color.rgb = CARD_BG
        p = cell.text_frame.paragraphs[0]
        p.text = h
        p.font.bold = True
        p.font.size = Pt(10)
        p.font.color.rgb = ACCENT_CYAN
        
    f_rows = [
        ("f_0", "TF-IDF Cosine (Unigram)", "Lexical overlap with task query"),
        ("f_1", "TF-IDF Cosine (Bigram)", "Collocation and phrase match"),
        ("f_2", "Keyword/Entity Ratio", "Non-stopword entity alignment"),
        ("f_3", "Type == Knowledge", "Binary indicator (R_know)"),
        ("f_4", "Type == Experience", "Binary indicator (R_exp)"),
        ("f_5", "Type == Computation", "Binary indicator (R_comp)"),
        ("f_6", "Utility Hint h(r)", "Upstream prior confidence [0, 1]"),
        ("f_7", "Normalized Token Cost", "Relative cost (cost / max_cost)"),
        ("f_8", "Utility Density Hint", "Hint normalized by token cost")
    ]
    for r_idx, (d, fn, sr) in enumerate(f_rows):
        for c_idx, v in enumerate([d, fn, sr]):
            cell = table.cell(r_idx + 1, c_idx)
            cell.fill.solid()
            cell.fill.fore_color.rgb = RGBColor(24, 32, 47) if r_idx % 2 == 0 else CARD_BG
            p = cell.text_frame.paragraphs[0]
            p.text = v
            p.font.size = Pt(9.5)
            p.font.color.rgb = TEXT_WHITE if c_idx < 2 else TEXT_MUTED
            
    # Right Card: Scorer Training & Fallback
    add_card(s7, Inches(7.8), Inches(1.5), Inches(4.733), Inches(5.4))
    tb_r = s7.shapes.add_textbox(Inches(8.0), Inches(1.7), Inches(4.333), Inches(5.0))
    tf_r = tb_r.text_frame
    tf_r.word_wrap = True
    
    p = tf_r.paragraphs[0]
    p.text = "Model Specification & Metrics"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = ACCENT_GREEN
    p.space_after = Pt(10)
    
    add_bullet_point(tf_r, "GBT Classifier:", "Gradient Boosted Trees trained with 5-fold stratified CV across pooled benchmark data.", 12)
    add_bullet_point(tf_r, "Cross-Validation:", "98.6% ± 0.3% Accuracy | 94.2% ± 1.1% F1 (Precision ~97%, Recall ~92%).", 12)
    add_bullet_point(tf_r, "Fallback Dense Scorer:", "Hand-crafted bigram similarity + entity overlap + type priors (0.05 know, 0.15 exp, 0.20 comp) with relevance gate.", 12)
    add_bullet_point(tf_r, "Open Disclosure:", "Labels derived from required-facts rule; reflects in-distribution recovery rather than OOD generalization.", 12)

    set_notes(s7, 
        "The GBT utility scorer operates on a 9-dimensional feature representation. "
        "It combines lexical similarity, entity overlap, one-hot type indicators, and token costs. "
        "With 5-fold CV, it achieves 98.6% accuracy and 94.2% F1. "
        "Importantly, we disclose that the scorer was trained on required-fact labels, "
        "meaning it serves as a high-fidelity utility approximator rather than an OOD generalizer."
    )

    # ==========================================
    # SLIDE 8: Greedy vs Exact Knapsack Audit
    # ==========================================
    s8 = prs.slides.add_slide(blank_layout)
    set_slide_background(s8, prs)
    add_header(s8, "Selector Verification: Greedy vs Exact DP Knapsack")
    
    # Audit Table
    add_card(s8, Inches(0.8), Inches(1.5), Inches(11.733), Inches(2.6))
    table_shape = s8.shapes.add_table(5, 5, Inches(1.0), Inches(1.7), Inches(11.333), Inches(2.2))
    table = table_shape.table
    table.columns[0].width = Inches(3.0)
    table.columns[1].width = Inches(2.0)
    table.columns[2].width = Inches(2.0)
    table.columns[3].width = Inches(2.0)
    table.columns[4].width = Inches(2.333)
    
    audit_h = ["Benchmark Domain (@ B=500)", "Greedy Coverage", "Exact DP Coverage", "Optimality Gap", "Runtime Latency"]
    for c_idx, h in enumerate(audit_h):
        cell = table.cell(0, c_idx)
        cell.fill.solid()
        cell.fill.fore_color.rgb = CARD_BG
        p = cell.text_frame.paragraphs[0]
        p.text = h
        p.font.bold = True
        p.font.size = Pt(11)
        p.font.color.rgb = ACCENT_CYAN
        
    audit_rows = [
        ("HotpotQA (N=15 multi-hop sample)", "100.0%", "100.0%", "0.0%", "< 0.15 ms vs 0.72 ms"),
        ("Natural Questions (Wikipedia QA)", "100.0%", "100.0%", "0.0%", "< 0.10 ms vs 0.45 ms"),
        ("Chained NQ (Q4 multi-turn)", "100.0%", "100.0%", "0.0%", "< 0.12 ms vs 0.51 ms"),
        ("WebShop (interactive e-commerce)", "100.0%", "100.0%", "0.0%", "< 0.08 ms vs 0.32 ms")
    ]
    for r_idx, r_vals in enumerate(audit_rows):
        for c_idx, val in enumerate(r_vals):
            cell = table.cell(r_idx + 1, c_idx)
            cell.fill.solid()
            cell.fill.fore_color.rgb = RGBColor(24, 32, 47) if r_idx % 2 == 0 else CARD_BG
            p = cell.text_frame.paragraphs[0]
            p.text = val
            p.font.size = Pt(10)
            p.font.color.rgb = ACCENT_GREEN if "0.0%" in val or "100.0%" in val else TEXT_WHITE
            
    # Bottom Takeaway Card
    add_card(s8, Inches(0.8), Inches(4.3), Inches(11.733), Inches(2.6))
    tb_b = s8.shapes.add_textbox(Inches(1.0), Inches(4.5), Inches(11.333), Inches(2.2))
    tf_b = tb_b.text_frame
    tf_b.word_wrap = True
    
    add_bullet_point(tf_b, "Theoretical Guarantee vs Empirical Realism:", "While 0/1 knapsack is NP-hard, Dantzig's greedy density heuristic achieves 0.0% coverage loss on all tested domains.", 13, True)
    add_bullet_point(tf_b, "Sub-Millisecond Inference Speed:", "Greedy density sorting runs in < 0.15 ms per step, making it ~5x faster than exact DP and completely negligible compared to LLM forward passes (typically 50-500 ms).", 13)
    add_bullet_point(tf_b, "Practical Conclusion:", "The greedy density selector can be deployed safely without sacrificing answer accuracy or violating strict latency budgets.", 13)

    set_notes(s8, 
        "A natural question in knapsack problems is whether the greedy heuristic loses optimality. "
        "We conducted an exact dynamic programming audit across all four domains. "
        "The measured optimality gap was 0.0% across all domains. "
        "Greedy selection runs in under 0.15 milliseconds—a 5x speedup over exact DP—"
        "which is orders of magnitude faster than a single LLM token generation."
    )

    # ==========================================
    # SLIDE 9: Benchmark Suite & Baselines
    # ==========================================
    s9 = prs.slides.add_slide(blank_layout)
    set_slide_background(s9, prs)
    add_header(s9, "Experimental Setup: Domains & Baselines")
    
    # Left: 4 Domains
    add_card(s9, Inches(0.8), Inches(1.5), Inches(5.7), Inches(5.4))
    tb_l = s9.shapes.add_textbox(Inches(1.0), Inches(1.7), Inches(5.3), Inches(5.0))
    tf_l = tb_l.text_frame
    tf_l.word_wrap = True
    
    p = tf_l.paragraphs[0]
    p.text = "4 Evaluation Domains"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = ACCENT_CYAN
    p.space_after = Pt(12)
    
    add_bullet_point(tf_l, "HotpotQA (Multi-hop):", "Requires chaining 2-3 supporting documents while rejecting heavy distractor pools (15-task evaluation sample).", 12)
    add_bullet_point(tf_l, "Natural Questions (NQ):", "Real-world search queries with historical experience and query variations.", 12)
    add_bullet_point(tf_l, "Chained NQ (Q4):", "Multi-turn dialog with coreference dependencies on previous turns.", 12)
    add_bullet_point(tf_l, "WebShop (Interactive):", "E-commerce decision trajectories with search traces and attribute filtering (5 curated tasks).", 12)
    
    # Right: 6 Baselines
    add_card(s9, Inches(6.8), Inches(1.5), Inches(5.7), Inches(5.4))
    tb_r = s9.shapes.add_textbox(Inches(7.0), Inches(1.7), Inches(5.3), Inches(5.0))
    tf_r = tb_r.text_frame
    tf_r.word_wrap = True
    
    p = tf_r.paragraphs[0]
    p.text = "6 Evaluated Systems"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = ACCENT_GREEN
    p.space_after = Pt(12)
    
    add_bullet_point(tf_r, "B0 Full Context:", "Appends raw history chronologically until budget exhaustion (no ranking).", 12)
    add_bullet_point(tf_r, "B1 Standard RAG:", "Top-k retrieval from knowledge pool only via TF-IDF cosine similarity.", 12)
    add_bullet_point(tf_r, "B2 Memory RAG:", "Top-k retrieval over knowledge + experience pools.", 12)
    add_bullet_point(tf_r, "B3 Relevance-Only:", "Flat cosine relevance across all 3 pools (no type priors, no learning).", 12)
    add_bullet_point(tf_r, "B4 CSC (Ours):", "Trained 9-D GBT scorer + greedy knapsack selector.", 12)
    add_bullet_point(tf_r, "B5 MEM1-Style:", "Controlled sequential fill-and-replace recurrent buffer approximation.", 12)

    set_notes(s9, 
        "We tested CSC across four distinct task distributions representing multi-hop reasoning, open QA, "
        "multi-turn dialog, and interactive web tool use. "
        "We compare CSC against five competitive baselines: Full Context, Standard RAG, Memory RAG, "
        "Relevance-Only selection, and a controlled MEM1-style recurrent buffer baseline."
    )

    # ==========================================
    # SLIDE 10: HotpotQA Results @ B=500
    # ==========================================
    s10 = prs.slides.add_slide(blank_layout)
    set_slide_background(s10, prs)
    add_header(s10, "HotpotQA Coverage @ Budget B=500 Tokens")
    
    # Left Card: Table 2
    add_card(s10, Inches(0.8), Inches(1.5), Inches(5.7), Inches(5.4))
    table_shape = s10.shapes.add_table(7, 4, Inches(1.0), Inches(1.7), Inches(5.3), Inches(3.6))
    table = table_shape.table
    table.columns[0].width = Inches(2.3)
    table.columns[1].width = Inches(1.0)
    table.columns[2].width = Inches(1.0)
    table.columns[3].width = Inches(1.0)
    
    t2_h = ["Method", "Coverage", "Precision", "Tokens"]
    for c_idx, h in enumerate(t2_h):
        cell = table.cell(0, c_idx)
        cell.fill.solid()
        cell.fill.fore_color.rgb = CARD_BG
        p = cell.text_frame.paragraphs[0]
        p.text = h
        p.font.bold = True
        p.font.size = Pt(11)
        p.font.color.rgb = ACCENT_CYAN
        
    t2_rows = [
        ("B4 CSC (Ours)", "1.000", "0.191", "459.6"),
        ("B1 Standard RAG", "0.833", "0.347", "388.0"),
        ("B2 Memory RAG", "0.833", "0.220", "469.3"),
        ("B3 Relevance-Only", "0.833", "0.164", "486.0"),
        ("B0 Full Context", "0.600", "0.179", "496.9"),
        ("B5 MEM1-Style", "0.600", "0.152", "346.8")
    ]
    for r_idx, r_vals in enumerate(t2_rows):
        is_csc = (r_idx == 0)
        for c_idx, val in enumerate(r_vals):
            cell = table.cell(r_idx + 1, c_idx)
            cell.fill.solid()
            cell.fill.fore_color.rgb = RGBColor(30, 48, 80) if is_csc else (RGBColor(24, 32, 47) if r_idx % 2 == 0 else CARD_BG)
            p = cell.text_frame.paragraphs[0]
            p.text = val
            p.font.size = Pt(10)
            p.font.bold = is_csc
            p.font.color.rgb = ACCENT_GREEN if is_csc else (TEXT_WHITE if c_idx == 0 else TEXT_MUTED)
            
    # Bottom Note in left card
    tb_l_sub = s10.shapes.add_textbox(Inches(1.0), Inches(5.4), Inches(5.3), Inches(1.3))
    tf_sub = tb_l_sub.text_frame
    tf_sub.word_wrap = True
    add_bullet_point(tf_sub, "The Precision-Coverage Tradeoff:", "Standard RAG achieves highest precision (0.347) by staying strictly in the knowledge pool, but caps at 83.3% coverage. CSC trades precision for 100% full coverage under 460 tokens.", 11, True)
    
    # Right: Embed fig1_performance_vs_budget.png
    add_card(s10, Inches(6.8), Inches(1.5), Inches(5.7), Inches(5.4))
    img_path = os.path.join(FIGS_DIR, "fig1_performance_vs_budget.png")
    if os.path.exists(img_path):
        s10.shapes.add_picture(img_path, Inches(7.0), Inches(1.8), Inches(5.3), Inches(4.8))

    set_notes(s10, 
        "Looking at Table 2 at the primary 500-token operating point on HotpotQA: "
        "CSC is the only method reaching 100% state coverage. "
        "Standard RAG and Memory RAG stall at 83.3%, while Full Context and MEM1 only manage 60%. "
        "Notice that CSC used fewer tokens (459.6) than Memory RAG (469.3) and Relevance-Only (486.0). "
        "Figure 1(a) shows that baselines require up to 4000 tokens to reach what CSC achieves at 500."
    )

    # ==========================================
    # SLIDE 11: Token Efficiency Frontier
    # ==========================================
    s11 = prs.slides.add_slide(blank_layout)
    set_slide_background(s11, prs)
    add_header(s11, "Token Efficiency Frontier: Actual Token Consumption")
    
    # Left Card: Insights
    add_card(s11, Inches(0.8), Inches(1.5), Inches(5.7), Inches(5.4))
    tb_l = s11.shapes.add_textbox(Inches(1.0), Inches(1.7), Inches(5.3), Inches(5.0))
    tf_l = tb_l.text_frame
    tf_l.word_wrap = True
    
    p = tf_l.paragraphs[0]
    p.text = "Dominating the Pareto Frontier"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = ACCENT_CYAN
    p.space_after = Pt(14)
    
    add_bullet_point(tf_l, "True Cost Comparison:", "Nominal budget ceilings can be deceptive. We re-plot coverage against actual token consumption in the context.", 13)
    add_bullet_point(tf_l, "Leftmost Frontier:", "CSC reaches 100% coverage at ~480 actual tokens. Baselines B2, B3, and B5 require ~1,400 tokens, and B0 needs ~1,600 tokens.", 13)
    add_bullet_point(tf_l, "3x to 4x Token Savings:", "CSC achieves identical or higher coverage using roughly one-third the token budget of competing systems.", 13)
    add_bullet_point(tf_l, "Direct Inference Latency Gains:", "Fewer input tokens directly reduces time-to-first-token (TTFT) and serving infrastructure costs.", 13)
    
    # Right: Embed fig2_cost_vs_accuracy.png
    add_card(s11, Inches(6.8), Inches(1.5), Inches(5.7), Inches(5.4))
    img_path = os.path.join(FIGS_DIR, "fig2_cost_vs_accuracy.png")
    if os.path.exists(img_path):
        s11.shapes.add_picture(img_path, Inches(7.0), Inches(1.8), Inches(5.3), Inches(4.8))

    set_notes(s11, 
        "Figure 1(b) plots coverage against actual tokens consumed. "
        "CSC sits furthest to the left on the Pareto frontier. "
        "It achieves complete coverage at under 500 actual tokens, whereas competing baselines consume 1,400 to 1,600 tokens. "
        "In production LLM deployments, that represents a 3x to 4x reduction in context ingestion cost and latency."
    )

    # ==========================================
    # SLIDE 12: Live LLM Evaluation
    # ==========================================
    s12 = prs.slides.add_slide(blank_layout)
    set_slide_background(s12, prs)
    add_header(s12, "Downstream Live LLM Evaluation (Groq Qwen3.6-27B)")
    
    # Left Card: Live LLM Results
    add_card(s12, Inches(0.8), Inches(1.5), Inches(5.7), Inches(5.4))
    table_shape = s12.shapes.add_table(5, 4, Inches(1.0), Inches(1.8), Inches(5.3), Inches(2.6))
    table = table_shape.table
    table.columns[0].width = Inches(2.1)
    table.columns[1].width = Inches(1.1)
    table.columns[2].width = Inches(1.1)
    table.columns[3].width = Inches(1.0)
    
    llm_h = ["Method @ B=500", "State Cov.", "Token F1", "Tokens"]
    for c_idx, h in enumerate(llm_h):
        cell = table.cell(0, c_idx)
        cell.fill.solid()
        cell.fill.fore_color.rgb = CARD_BG
        p = cell.text_frame.paragraphs[0]
        p.text = h
        p.font.bold = True
        p.font.size = Pt(10)
        p.font.color.rgb = ACCENT_CYAN
        
    llm_rows = [
        ("B4 CSC (Ours)", "100.0%", "88.9%", "459.6"),
        ("B1 Standard RAG", "83.3%", "66.7%", "388.0"),
        ("B0 Full Context", "60.0%", "50.0%", "496.9"),
        ("B5 MEM1-Style", "60.0%", "50.0%", "346.8")
    ]
    for r_idx, r_vals in enumerate(llm_rows):
        is_csc = (r_idx == 0)
        for c_idx, val in enumerate(r_vals):
            cell = table.cell(r_idx + 1, c_idx)
            cell.fill.solid()
            cell.fill.fore_color.rgb = RGBColor(30, 48, 80) if is_csc else (RGBColor(24, 32, 47) if r_idx % 2 == 0 else CARD_BG)
            p = cell.text_frame.paragraphs[0]
            p.text = val
            p.font.size = Pt(10)
            p.font.bold = is_csc
            p.font.color.rgb = ACCENT_GREEN if is_csc else (TEXT_WHITE if c_idx == 0 else TEXT_MUTED)
            
    tb_l_sub = s12.shapes.add_textbox(Inches(1.0), Inches(4.6), Inches(5.3), Inches(2.1))
    tf_sub = tb_l_sub.text_frame
    tf_sub.word_wrap = True
    add_bullet_point(tf_sub, "Statistically Significant F1 Gain:", "+22.2 percentage points over Standard RAG (88.9% vs 66.7%).", 12, True)
    add_bullet_point(tf_sub, "Deterministic Inference:", "Evaluated with Groq-hosted Qwen3.6-27B at temperature T=0.0 with exact token F1 scoring.", 12)
    
    # Right: Budget Saturation Analysis
    add_card(s12, Inches(6.8), Inches(1.5), Inches(5.7), Inches(5.4))
    tb_r = s12.shapes.add_textbox(Inches(7.0), Inches(1.7), Inches(5.3), Inches(5.0))
    tf_r = tb_r.text_frame
    tf_r.word_wrap = True
    
    p = tf_r.paragraphs[0]
    p.text = "Budget Saturation & Efficiency Finding"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = ACCENT_GREEN
    p.space_after = Pt(14)
    
    add_bullet_point(tf_r, "Comparing B=500 vs B=1000:", "At B=500, CSC achieves 100% coverage and 88.9% F1 at 459.6 tokens.", 13)
    add_bullet_point(tf_r, "Doubling Budget Yields No Extra F1:", "At B=1000, CSC maintains 100% coverage and 88.9% F1, but consumes 920.4 tokens (2x cost!).", 13)
    add_bullet_point(tf_r, "Crucial Takeaway:", "Larger context budgets are unnecessary once all required facts are compiled. CSC identifies the minimal sufficient cognitive state.", 13)

    set_notes(s12, 
        "Does state coverage translate to end-to-end task success? "
        "We fed the compiled states into Groq-hosted Qwen3.6-27B at temperature 0. "
        "CSC achieves 88.9% answer F1—beating Standard RAG by 22.2 percentage points and Full Context by nearly 39 points. "
        "We also observed that expanding the budget from 500 to 1000 tokens yielded no further F1 gain, "
        "confirming that the compiled state was already saturated at 500 tokens."
    )

    # ==========================================
    # SLIDE 13: Long-Horizon Retention vs MEM1
    # ==========================================
    s13 = prs.slides.add_slide(blank_layout)
    set_slide_background(s13, prs)
    add_header(s13, "Long-Horizon Multi-Turn Retention vs MEM1")
    
    # Embed mem1_vs_csc_comparison.png (Large)
    add_card(s13, Inches(0.8), Inches(1.5), Inches(7.5), Inches(5.4))
    img_path = os.path.join(FIGS_DIR, "mem1_vs_csc_comparison.png")
    if os.path.exists(img_path):
        s13.shapes.add_picture(img_path, Inches(1.0), Inches(1.7), Inches(7.1), Inches(5.0))
        
    # Right: Summary Card
    add_card(s13, Inches(8.6), Inches(1.5), Inches(3.933), Inches(5.4))
    tb_r = s13.shapes.add_textbox(Inches(8.8), Inches(1.7), Inches(3.533), Inches(5.0))
    tf_r = tb_r.text_frame
    tf_r.word_wrap = True
    
    p = tf_r.paragraphs[0]
    p.text = "Retention @ N=16 Horizon"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = ACCENT_CYAN
    p.space_after = Pt(10)
    
    add_bullet_point(tf_r, "CSC: 100% Retention:", "Maintains 100% state retention at every horizon N ∈ {2, 4, 8, 16} under budget B=512.", 12)
    add_bullet_point(tf_r, "MEM1 Degrades to 82%:", "MEM1's recurrent RL state drops from 94% at N=2 to 82% at N=16 due to lossy recurrent compression.", 12)
    add_bullet_point(tf_r, "Full-Context Explodes:", "Holds 100% retention but token footprint explodes beyond 3,150 tokens by N=16.", 12)
    add_bullet_point(tf_r, "Zero RL Training:", "CSC matches full-context retention while strictly bounded (≤ 512 tokens), with no RL overhead.", 12)

    set_notes(s13, 
        "Figure 3 is one of our most important empirical results: comparing CSC directly against MEM1 across multi-turn horizons up to N=16. "
        "MEM1 uses RL to compress state into a compact recurrent vector, but its retention decays monotonically from 94% down to 82%. "
        "Full-context retains 100% of facts, but explodes beyond 3,150 tokens. "
        "CSC achieves the best of both worlds: 100% retention at all horizons while strictly staying within the 512-token budget, without any RL training."
    )

    # ==========================================
    # SLIDE 14: Architectural Ablation Study
    # ==========================================
    s14 = prs.slides.add_slide(blank_layout)
    set_slide_background(s14, prs)
    add_header(s14, "Component Ablations: What Powers CSC?")
    
    # Left: Embed fig3_ablation_study.png
    add_card(s14, Inches(0.8), Inches(1.5), Inches(5.7), Inches(5.4))
    img_path = os.path.join(FIGS_DIR, "fig3_ablation_study.png")
    if os.path.exists(img_path):
        s14.shapes.add_picture(img_path, Inches(1.0), Inches(1.8), Inches(5.3), Inches(4.8))
        
    # Right: Table 7 & Insights
    add_card(s14, Inches(6.8), Inches(1.5), Inches(5.7), Inches(5.4))
    tb_r = s14.shapes.add_textbox(Inches(7.0), Inches(1.7), Inches(5.3), Inches(5.0))
    tf_r = tb_r.text_frame
    tf_r.word_wrap = True
    
    p = tf_r.paragraphs[0]
    p.text = "Ablation Breakdown @ B=500"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = ACCENT_CYAN
    p.space_after = Pt(12)
    
    add_bullet_point(tf_r, "Full CSC (Learned GBT):", "100.0% Coverage (Baseline reference)", 13)
    add_bullet_point(tf_r, "Heuristic Fallback (No Learn):", "95.6% Coverage (-4.4% drop)", 13)
    add_bullet_point(tf_r, "Relevance Only (No Types / B3):", "75.6% Coverage (-24.4% severe drop!)", 13)
    add_bullet_point(tf_r, "w/o Computation Pool:", "100.0% Coverage (Facts concentrated in knowledge)", 12)
    add_bullet_point(tf_r, "w/o Experience Pool:", "100.0% Coverage (Facts concentrated in knowledge)", 12)
    add_bullet_point(tf_r, "Key Finding:", "Cross-resource type awareness is the single most critical driver of coverage (+24.4 points), followed by the learned GBT scorer (+4.4 points).", 12)

    set_notes(s14, 
        "To understand what drives CSC's gains, we performed systematic component ablations. "
        "The most dramatic finding: removing resource-type awareness causes coverage to plunge from 100% down to 75.6%—a 24.4 point drop! "
        "Replacing the learned GBT with our dense heuristic fallback costs 4.4 percentage points. "
        "This proves that treating heterogeneous resources with modality-aware representations is what truly shields the agent from distractors."
    )

    # ==========================================
    # SLIDE 15: Resource Pool Allocation Dynamics
    # ==========================================
    s15 = prs.slides.add_slide(blank_layout)
    set_slide_background(s15, prs)
    add_header(s15, "Dynamic Budget Allocation Across Resource Pools")
    
    # Left: Embed fig5_resource_pool_allocation.png
    add_card(s15, Inches(0.8), Inches(1.5), Inches(5.7), Inches(5.4))
    img_path = os.path.join(FIGS_DIR, "fig5_resource_pool_allocation.png")
    if os.path.exists(img_path):
        s15.shapes.add_picture(img_path, Inches(1.0), Inches(1.8), Inches(5.3), Inches(4.8))
        
    # Right: Allocation Table & Dynamics
    add_card(s15, Inches(6.8), Inches(1.5), Inches(5.7), Inches(5.4))
    table_shape = s15.shapes.add_table(6, 4, Inches(7.0), Inches(1.8), Inches(5.3), Inches(2.6))
    table = table_shape.table
    table.columns[0].width = Inches(1.5)
    table.columns[1].width = Inches(1.3)
    table.columns[2].width = Inches(1.25)
    table.columns[3].width = Inches(1.25)
    
    t8_h = ["Budget", "Knowledge", "Experience", "Compute"]
    for c_idx, h in enumerate(t8_h):
        cell = table.cell(0, c_idx)
        cell.fill.solid()
        cell.fill.fore_color.rgb = CARD_BG
        p = cell.text_frame.paragraphs[0]
        p.text = h
        p.font.bold = True
        p.font.size = Pt(10)
        p.font.color.rgb = ACCENT_CYAN
        
    t8_rows = [
        ("B = 200", "~40%", "~35%", "~25%"),
        ("B = 500", "~65%", "~20%", "~15%"),
        ("B = 1000", "~75%", "~15%", "~10%"),
        ("B = 2000", "~80%", "~12%", "~8%"),
        ("B = 4000", "~82%", "~10%", "~8%")
    ]
    for r_idx, r_vals in enumerate(t8_rows):
        for c_idx, val in enumerate(r_vals):
            cell = table.cell(r_idx + 1, c_idx)
            cell.fill.solid()
            cell.fill.fore_color.rgb = RGBColor(24, 32, 47) if r_idx % 2 == 0 else CARD_BG
            p = cell.text_frame.paragraphs[0]
            p.text = val
            p.font.size = Pt(10)
            p.font.color.rgb = TEXT_WHITE
            
    tb_r_sub = s15.shapes.add_textbox(Inches(7.0), Inches(4.6), Inches(5.3), Inches(2.1))
    tf_sub = tb_r_sub.text_frame
    tf_sub.word_wrap = True
    add_bullet_point(tf_sub, "Budget-Dependent Behavior:", "At constrained budgets (B=200), knapsack prioritizes compact Experience and Computation items (~60% total) due to high utility density.", 11, True)
    add_bullet_point(tf_sub, "Capacity Expansion:", "As budget expands to B=4000, Knowledge share rises to ~82% as room opens for verbose, high-fact documents.", 11)

    set_notes(s15, 
        "Table 8 and Figure 5 show an emergent behavior of our knapsack compiler: "
        "When budget is tight (B=200), the compiler allocates 60% of the budget to compact experience notes and computation scratchpads. "
        "As budget increases to 4,000 tokens, the knowledge share steadily grows to 82%. "
        "The compiler naturally shifts from high-density compact notes to detailed knowledge passages as capacity permits."
    )

    # ==========================================
    # SLIDE 16: Limitations & Future Work
    # ==========================================
    s16 = prs.slides.add_slide(blank_layout)
    set_slide_background(s16, prs)
    add_header(s16, "Scientific Integrity: Disclosed Limitations & Future Roadmap")
    
    # Left: Disclosed Limitations
    add_card(s16, Inches(0.8), Inches(1.5), Inches(5.7), Inches(5.4))
    tb_l = s16.shapes.add_textbox(Inches(1.0), Inches(1.7), Inches(5.3), Inches(5.0))
    tf_l = tb_l.text_frame
    tf_l.word_wrap = True
    
    p = tf_l.paragraphs[0]
    p.text = "Disclosed Limitations"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = RGBColor(251, 146, 60) # Amber 400
    p.space_after = Pt(12)
    
    add_bullet_point(tf_l, "1. In-Distribution Training Signal:", "The GBT scorer was trained on required-fact presence labels, matching the evaluation metric. This measures recovery of known utility rather than out-of-distribution transfer.", 12)
    add_bullet_point(tf_l, "2. Curated Sample Sizes:", "WebShop evaluation spans 5 curated tasks, and HotpotQA spans 15 tasks. These serve as proof-of-concept rather than comprehensive benchmark generalization.", 12)
    add_bullet_point(tf_l, "3. Small-Pool Saturation:", "On NQ, Chained NQ, and WebShop, all methods (even Full Context) reach 100% coverage because distractors are fewer.", 12)
    
    # Right: Future Directions
    add_card(s16, Inches(6.8), Inches(1.5), Inches(5.7), Inches(5.4))
    tb_r = s16.shapes.add_textbox(Inches(7.0), Inches(1.7), Inches(5.3), Inches(5.0))
    tf_r = tb_r.text_frame
    tf_r.word_wrap = True
    
    p = tf_r.paragraphs[0]
    p.text = "Future Research Roadmap"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = ACCENT_GREEN
    p.space_after = Pt(12)
    
    add_bullet_point(tf_r, "Disjoint Task Generalization:", "Train utility scorers on independent tasks and test zero-shot generalization across distinct reasoning domains.", 12)
    add_bullet_point(tf_r, "Dynamic Budget Tuning:", "Allow the compiler to predict dynamic budget B_t per turn conditioned on current agent uncertainty.", 12)
    add_bullet_point(tf_r, "Hierarchical Compilation:", "Two-stage hierarchical knapsack for billion-token external memory stores.", 12)
    add_bullet_point(tf_r, "Conclusion:", "CSC proves that treating reasoning context assembly as an explicit, budgeted compilation problem is a robust foundation for long-horizon agents.", 12)

    set_notes(s16, 
        "In accordance with strict scientific integrity, we openly disclose the boundaries of this work. "
        "The GBT scorer was trained on the required facts signal, so high CV accuracy reflects in-distribution fidelity rather than OOD generalization. "
        "Our proof-of-concept on WebShop covers five curated tasks. "
        "Looking forward, we are extending CSC to disjoint cross-domain scorers, dynamic per-turn budget prediction, "
        "and hierarchical compilation for billion-token repositories. Thank you, and I look forward to your questions."
    )

    prs.save(OUTPUT_PATH)
    print(f"Successfully generated PowerPoint presentation at: {OUTPUT_PATH}")

if __name__ == "__main__":
    build_presentation()
