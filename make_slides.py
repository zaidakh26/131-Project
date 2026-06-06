from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt
import copy

# ── theme colors ──────────────────────────────────────────────────────────────
BG      = RGBColor(0x0D, 0x1B, 0x2A)   # dark navy
ACCENT  = RGBColor(0x00, 0xB4, 0xD8)   # cyan
WHITE   = RGBColor(0xFF, 0xFF, 0xFF)
GRAY    = RGBColor(0xAA, 0xBB, 0xCC)
GREEN   = RGBColor(0x06, 0xD6, 0xA0)
YELLOW  = RGBColor(0xFF, 0xD1, 0x66)

W = Inches(13.33)
H = Inches(7.5)

prs = Presentation()
prs.slide_width  = W
prs.slide_height = H

def blank_slide(prs):
    layout = prs.slide_layouts[6]  # blank
    return prs.slides.add_slide(layout)

def bg(slide):
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = BG

def add_text(slide, text, left, top, width, height,
             size=24, bold=False, color=WHITE, align=PP_ALIGN.LEFT, italic=False):
    txBox = slide.shapes.add_textbox(left, top, width, height)
    tf = txBox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = color
    return txBox

def add_bullet_box(slide, items, left, top, width, height, size=20, color=WHITE, title_color=ACCENT):
    txBox = slide.shapes.add_textbox(left, top, width, height)
    tf = txBox.text_frame
    tf.word_wrap = True
    for i, item in enumerate(items):
        p = tf.add_paragraph() if i > 0 else tf.paragraphs[0]
        p.space_before = Pt(4)
        run = p.add_run()
        run.text = item
        run.font.size = Pt(size)
        run.font.color.rgb = color
    return txBox

def accent_bar(slide, top=Inches(0.55)):
    bar = slide.shapes.add_shape(1, 0, top, W, Pt(3))
    bar.fill.solid()
    bar.fill.fore_color.rgb = ACCENT
    bar.line.fill.background()

def stat_box(slide, value, label, left, top, w=Inches(2.8), h=Inches(1.4)):
    box = slide.shapes.add_shape(1, left, top, w, h)
    box.fill.solid()
    box.fill.fore_color.rgb = RGBColor(0x1A, 0x2E, 0x44)
    box.line.color.rgb = ACCENT
    add_text(slide, value, left + Inches(0.1), top + Inches(0.05), w - Inches(0.2), Inches(0.7),
             size=32, bold=True, color=ACCENT, align=PP_ALIGN.CENTER)
    add_text(slide, label, left + Inches(0.1), top + Inches(0.75), w - Inches(0.2), Inches(0.55),
             size=14, color=GRAY, align=PP_ALIGN.CENTER)

# ═══════════════════════════════════════════════════════════════════════════════
# SLIDE 1 — Title
# ═══════════════════════════════════════════════════════════════════════════════
s = blank_slide(prs)
bg(s)
accent_bar(s, top=Inches(3.8))

add_text(s, "Real-Time Exercise Form Analysis",
         Inches(0.8), Inches(1.2), Inches(11.5), Inches(1.2),
         size=44, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
add_text(s, "Using Pose Estimation",
         Inches(0.8), Inches(2.3), Inches(11.5), Inches(0.8),
         size=36, bold=False, color=ACCENT, align=PP_ALIGN.CENTER)
add_text(s, "Zaid Akhtar  ·  Heena Khan",
         Inches(0.8), Inches(4.2), Inches(11.5), Inches(0.6),
         size=22, color=GRAY, align=PP_ALIGN.CENTER)
add_text(s, "CS 131  ·  Demo Day",
         Inches(0.8), Inches(4.8), Inches(11.5), Inches(0.5),
         size=18, color=GRAY, align=PP_ALIGN.CENTER)

# ═══════════════════════════════════════════════════════════════════════════════
# SLIDE 2 — Problem & Motivation
# ═══════════════════════════════════════════════════════════════════════════════
s = blank_slide(prs)
bg(s)
accent_bar(s)
add_text(s, "Problem & Motivation", Inches(0.5), Inches(0.1), Inches(10), Inches(0.6),
         size=30, bold=True, color=ACCENT)

add_bullet_box(s, [
    "⚠️  ~3.5M gym injuries per year in the U.S. — many from poor lifting form",
    "💸  Personal trainers are expensive and not always available",
    "📷  Smartphones + webcams are everywhere — why not use them?",
    "",
    "→  Goal: build a real-time coaching system that detects bad squat",
    "    and RDL form using only a camera, no wearables",
], Inches(0.6), Inches(0.9), Inches(12), Inches(4.5), size=22)

add_text(s, "Exercises targeted:  Squat  ·  Romanian Deadlift (RDL)",
         Inches(0.6), Inches(5.5), Inches(12), Inches(0.6),
         size=20, bold=True, color=YELLOW)

# ═══════════════════════════════════════════════════════════════════════════════
# SLIDE 3 — Methodology
# ═══════════════════════════════════════════════════════════════════════════════
s = blank_slide(prs)
bg(s)
accent_bar(s)
add_text(s, "Methodology", Inches(0.5), Inches(0.1), Inches(10), Inches(0.6),
         size=30, bold=True, color=ACCENT)

# left column
add_text(s, "Pipeline", Inches(0.6), Inches(0.85), Inches(5.5), Inches(0.45),
         size=20, bold=True, color=YELLOW)
add_bullet_box(s, [
    "1.  Webcam / video input  →  OpenCV",
    "2.  Pose landmark detection  →  MediaPipe BlazePose",
    "       (33 3-D body keypoints per frame)",
    "3.  Joint angle computation",
    "       Knee angle:  hip – knee – ankle",
    "       Hip angle:   shoulder – hip – knee",
    "4.  EMA smoothing  (α = 0.2)  to reduce jitter",
    "5.  Threshold-based feedback + rep counting",
], Inches(0.6), Inches(1.4), Inches(6), Inches(4.5), size=18)

# right column
add_text(s, "Feedback Rules (Squat)", Inches(7.2), Inches(0.85), Inches(5.5), Inches(0.45),
         size=20, bold=True, color=YELLOW)
add_bullet_box(s, [
    "Torso lean   →  torso_lean  > 108°",
    "Knee cave    →  lateral dev > 0.10",
    "Go deeper    →  knee angle  > 60°",
    "Good depth  ←  none of above",
    "",
    "Feedback Rules (RDL)",
    "Too much knee bend  →  min knee < 110°",
    "Go deeper           →  min hip  > 100°",
    "Good                ←  none of above",
], Inches(7.2), Inches(1.4), Inches(5.7), Inches(4.5), size=18)

# ═══════════════════════════════════════════════════════════════════════════════
# SLIDE 4 — Results
# ═══════════════════════════════════════════════════════════════════════════════
s = blank_slide(prs)
bg(s)
accent_bar(s)
add_text(s, "Results", Inches(0.5), Inches(0.1), Inches(10), Inches(0.6),
         size=30, bold=True, color=ACCENT)

# stat boxes
stat_box(s, "42.9%",  "Squat accuracy\n31,628 frames (Kaggle)",  Inches(0.5),  Inches(1.1))
stat_box(s, "67%",    "Squat clip accuracy\n3 side-view clips",   Inches(3.6),  Inches(1.1))
stat_box(s, "60%",    "RDL clip accuracy\n5 side-view clips",     Inches(6.7),  Inches(1.1))
stat_box(s, "2",      "Exercises supported\nSquat + RDL",         Inches(9.8),  Inches(1.1))

# per-class table
add_text(s, "Squat — Per-class F1  (Kaggle frame-level eval)", Inches(0.6), Inches(2.75),
         Inches(12), Inches(0.45), size=18, bold=True, color=YELLOW)

headers = ["Class", "Precision", "Recall", "F1"]
rows = [
    ["torso_lean",          "0.534", "0.599", "0.565"],
    ["knee_cave",           "0.422", "0.576", "0.487"],
    ["go_deeper",           "0.343", "0.394", "0.367"],
    ["good_depth",          "0.407", "0.148", "0.217"],
]
col_x = [Inches(0.6), Inches(3.5), Inches(6.0), Inches(8.5)]
col_w = Inches(2.7)
row_h = Inches(0.38)
for ci, h in enumerate(headers):
    add_text(s, h, col_x[ci], Inches(3.25), col_w, row_h, size=16, bold=True, color=ACCENT)
for ri, row in enumerate(rows):
    y = Inches(3.65) + ri * row_h
    rc = GREEN if float(row[3]) > 0.45 else (YELLOW if float(row[3]) > 0.3 else WHITE)
    for ci, cell in enumerate(row):
        add_text(s, cell, col_x[ci], y, col_w, row_h, size=16,
                 color=rc if ci > 0 else WHITE)

add_text(s, "Key insight: torso lean & knee cave are easiest to detect; good_depth hardest (default class)",
         Inches(0.6), Inches(5.85), Inches(12), Inches(0.5), size=16, italic=True, color=GRAY)

# ═══════════════════════════════════════════════════════════════════════════════
# SLIDE 5 — Demo
# ═══════════════════════════════════════════════════════════════════════════════
s = blank_slide(prs)
bg(s)
accent_bar(s)
add_text(s, "Live Demo", Inches(0.5), Inches(0.1), Inches(10), Inches(0.6),
         size=30, bold=True, color=ACCENT)

add_bullet_box(s, [
    "🎥  Run squat_tracker.py on webcam  (side view, full body in frame)",
    "",
    "What you'll see:",
    "   •  MediaPipe skeleton overlay on live video",
    "   •  Real-time knee angle + hip angle displayed",
    "   •  Rep counter updates automatically",
    "   •  Feedback cues:  'Go deeper'  /  'Knee cave'  /  'Good depth'",
    "",
    "Camera tip:  place camera to your side, 6–8 ft away,",
    "                  so your full body (head → feet) is visible",
], Inches(0.8), Inches(1.0), Inches(11.5), Inches(5.5), size=22)

# ═══════════════════════════════════════════════════════════════════════════════
# SLIDE 6 — Learnings & Future Work
# ═══════════════════════════════════════════════════════════════════════════════
s = blank_slide(prs)
bg(s)
accent_bar(s)
add_text(s, "Learnings & Future Work", Inches(0.5), Inches(0.1), Inches(10), Inches(0.6),
         size=30, bold=True, color=ACCENT)

add_text(s, "What We Learned", Inches(0.6), Inches(0.85), Inches(5.8), Inches(0.45),
         size=22, bold=True, color=YELLOW)
add_bullet_box(s, [
    "✓  Camera angle is the #1 factor — side view essential",
    "✓  EMA smoothing critical for stable rep detection",
    "✓  Rule-based thresholds work but need per-user tuning",
    "✓  Clip-level eval harder than frame-level eval",
    "✓  No public RDL form dataset exists — we built our own eval set",
], Inches(0.6), Inches(1.4), Inches(6), Inches(3.5), size=20)

add_text(s, "Future Work", Inches(7.2), Inches(0.85), Inches(5.8), Inches(0.45),
         size=22, bold=True, color=YELLOW)
add_bullet_box(s, [
    "→  ML classifier (SVM / small NN) to replace thresholds",
    "→  Bench press support",
    "→  Automatic camera angle detection",
    "→  Mobile app deployment",
    "→  Multi-person tracking",
], Inches(7.2), Inches(1.4), Inches(5.7), Inches(3.5), size=20)

add_text(s, "Thank you!  Questions?",
         Inches(0.5), Inches(5.8), Inches(12.3), Inches(0.8),
         size=28, bold=True, color=ACCENT, align=PP_ALIGN.CENTER)

# ── save ──────────────────────────────────────────────────────────────────────
out = "/Users/henakhan/131-Project/CS131_Demo_Day.pptx"
prs.save(out)
print(f"Saved: {out}")
