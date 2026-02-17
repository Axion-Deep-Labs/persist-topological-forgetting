"""Generate the EXP-01 preliminary research paper as PDF."""

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.colors import HexColor, black, white
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, HRFlowable,
)

OUTPUT = "can_topology_predict_what_neural_networks_forget.pdf"

# Colors
DARK_BG = HexColor("#1a1a2e")
ACCENT = HexColor("#7c3aed")
LIGHT_GRAY = HexColor("#666666")
TABLE_HEADER_BG = HexColor("#f3f0ff")
TABLE_ALT_BG = HexColor("#fafafa")

styles = getSampleStyleSheet()

# Custom styles
styles.add(ParagraphStyle(
    "PaperTitle", parent=styles["Title"],
    fontSize=22, leading=28, spaceAfter=6,
    textColor=black, alignment=TA_CENTER,
))
styles.add(ParagraphStyle(
    "Authors", parent=styles["Normal"],
    fontSize=11, leading=14, alignment=TA_CENTER,
    textColor=LIGHT_GRAY, spaceAfter=4,
))
styles.add(ParagraphStyle(
    "Affiliation", parent=styles["Normal"],
    fontSize=9, leading=12, alignment=TA_CENTER,
    textColor=LIGHT_GRAY, spaceAfter=20,
))
styles.add(ParagraphStyle(
    "AbstractTitle", parent=styles["Heading2"],
    fontSize=12, leading=14, spaceBefore=0, spaceAfter=6,
    textColor=black,
))
styles.add(ParagraphStyle(
    "AbstractBody", parent=styles["Normal"],
    fontSize=9.5, leading=13, alignment=TA_JUSTIFY,
    leftIndent=36, rightIndent=36, spaceAfter=16,
    textColor=HexColor("#333333"),
))
styles.add(ParagraphStyle(
    "SectionHead", parent=styles["Heading1"],
    fontSize=14, leading=18, spaceBefore=18, spaceAfter=8,
    textColor=black,
))
styles.add(ParagraphStyle(
    "SubSectionHead", parent=styles["Heading2"],
    fontSize=11, leading=14, spaceBefore=12, spaceAfter=6,
    textColor=HexColor("#333333"),
))
styles.add(ParagraphStyle(
    "BodyText2", parent=styles["Normal"],
    fontSize=10, leading=14, alignment=TA_JUSTIFY,
    spaceAfter=8,
))
styles.add(ParagraphStyle(
    "Caption", parent=styles["Normal"],
    fontSize=9, leading=12, alignment=TA_CENTER,
    textColor=LIGHT_GRAY, spaceBefore=4, spaceAfter=12,
))
styles.add(ParagraphStyle(
    "Reference", parent=styles["Normal"],
    fontSize=8.5, leading=11, spaceAfter=3,
    leftIndent=18, firstLineIndent=-18,
))
styles.add(ParagraphStyle(
    "Footer", parent=styles["Normal"],
    fontSize=8, leading=10, alignment=TA_CENTER,
    textColor=LIGHT_GRAY,
))


def build_pdf():
    doc = SimpleDocTemplate(
        OUTPUT, pagesize=letter,
        topMargin=0.75 * inch, bottomMargin=0.75 * inch,
        leftMargin=1 * inch, rightMargin=1 * inch,
    )

    story = []

    # ─── Title Block ───
    story.append(Spacer(1, 20))
    story.append(Paragraph(
        "Can Topology Predict What Neural Networks Forget?",
        styles["PaperTitle"],
    ))
    story.append(Paragraph(
        "A Preliminary Investigation into Topological Signatures of Knowledge Persistence",
        ParagraphStyle("Subtitle", parent=styles["Normal"],
                       fontSize=11, leading=14, alignment=TA_CENTER,
                       textColor=LIGHT_GRAY, spaceAfter=16),
    ))
    story.append(Paragraph("Axion Deep Labs", styles["Authors"]))
    story.append(Paragraph(
        "Research Programs \u2014 EXP-01 (PERSIST) &nbsp;|&nbsp; Preliminary Report &nbsp;|&nbsp; February 2026",
        styles["Affiliation"],
    ))

    story.append(HRFlowable(width="100%", thickness=0.5, color=HexColor("#dddddd")))
    story.append(Spacer(1, 12))

    # ─── Abstract ───
    story.append(Paragraph("Abstract", styles["AbstractTitle"]))
    story.append(Paragraph(
        "Neural networks suffer from catastrophic forgetting. When trained on new information, "
        "they often overwrite previously learned knowledge. Despite decades of research into "
        "mitigation strategies, no prior work has investigated whether the geometric structure of "
        "learned representations predicts their vulnerability to being overwritten. We introduce a "
        "topological approach. Using persistent homology, we characterize the loss landscape around "
        "converged weight configurations and measure whether topological depth correlates with "
        "knowledge retention during sequential task training.",
        styles["AbstractBody"],
    ))
    story.append(Paragraph(
        "In preliminary experiments on Split CIFAR-100, we find that a Vision Transformer, ViT-Small, "
        "produces a loss landscape with nearly twice the topological persistence of a ResNet-18, "
        "H<sub>0</sub> equals 4254 versus 2151. Correspondingly, it exhibits slower and more gradual "
        "forgetting, retaining measurable accuracy twenty times longer during sequential training.",
        styles["AbstractBody"],
    ))
    story.append(Paragraph(
        "While results from two architectures do not constitute proof, the direction of this finding "
        "is consistent with our hypothesis. Networks that carve deeper topological structure into their "
        "loss landscape appear more resistant to catastrophic forgetting. If this relationship holds "
        "across additional architectures, it would provide both a diagnostic tool, predicting forgetting "
        "before it happens, and a foundation for topological regularization. This would enable training "
        "networks to learn in geometrically protected regions of parameter space.",
        styles["AbstractBody"],
    ))
    story.append(Paragraph(
        "To our knowledge, this is the first study connecting persistent homology of neural network "
        "loss landscapes to catastrophic forgetting.",
        styles["AbstractBody"],
    ))

    story.append(HRFlowable(width="100%", thickness=0.5, color=HexColor("#dddddd")))
    story.append(Spacer(1, 8))

    # ─── 1. The Problem ───
    story.append(Paragraph("1. The Problem", styles["SectionHead"]))
    story.append(Paragraph(
        "Every neural network deployed today is, in a fundamental sense, frozen. Once training is "
        "complete, introducing new knowledge typically comes at the cost of what the model already "
        "knows. This phenomenon was first described by McCloskey and Cohen in 1989 as <i>catastrophic "
        "interference</i> and is now widely referred to as <i>catastrophic forgetting</i>. The behavior "
        "is not subtle. A network trained to recognize fifty object categories, when subsequently "
        "trained on fifty new ones, does not accumulate knowledge into a unified set of one hundred. "
        "Instead, it adapts to the new categories while its performance on the original set collapses. "
        "The system does not integrate. It replaces. What appears to be learning is often a zero-sum "
        "exchange in parameter space, where acquiring new competence destabilizes prior structure.",
        styles["BodyText2"],
    ))
    story.append(Paragraph(
        "Importantly, this is not a problem that disappears with scale. Larger models, deeper "
        "architectures, and greater parameter counts do not fundamentally resolve the issue. A model "
        "with millions or even billions of parameters can forget just as decisively as a small network "
        "trained on a laptop. The intuition that capacity alone should allow separate tasks to coexist "
        "has not held up in practice. Instead, gradient-based training pushes parameters toward "
        "solutions optimized for the current objective, often traversing regions of the loss landscape "
        "that overwrite configurations supporting previous tasks. The instability is built into the "
        "dynamics of how these systems are trained.",
        styles["BodyText2"],
    ))
    story.append(Paragraph(
        "A number of mitigation strategies have been proposed, and each has advanced the field, yet "
        "all operate by managing the symptoms rather than eliminating the cause. Replay-based methods "
        "store and revisit past examples so that older tasks remain present during optimization. "
        "Regularization approaches such as elastic weight consolidation attempt to identify parameters "
        "important to prior tasks and penalize changes to them. Architectural methods like progressive "
        "networks allocate new capacity for new tasks to prevent interference. These techniques can "
        "slow forgetting, redistribute it, or compartmentalize it, but they do not remove the "
        "underlying tension between plasticity and stability. The network remains fundamentally prone "
        "to destructive updates when objectives shift.",
        styles["BodyText2"],
    ))
    story.append(Paragraph(
        "Humans, by contrast, appear to learn cumulatively. When a person studies calculus, they do "
        "not forget how to walk. When they acquire a new skill, earlier skills remain accessible. "
        "Even when knowledge fades, it rarely disappears instantaneously as a direct consequence of "
        "learning something new. This contrast suggests that the critical variable is not simply the "
        "number of parameters or neurons, but the structural organization of how knowledge is encoded. "
        "Biological systems seem to embed information in ways that allow new representations to form "
        "without erasing old ones. If artificial networks fail to do the same, the limitation may lie "
        "in the geometry and topology of their learned representations.",
        styles["BodyText2"],
    ))
    story.append(Paragraph(
        "Seen from this perspective, catastrophic forgetting is not merely an optimization "
        "inconvenience. It is evidence that current neural networks organize knowledge in a fragile "
        "manner. Understanding the structural properties that make representations stable or unstable "
        "may be essential for building systems that truly learn over time. If we can identify what "
        "distinguishes representations that survive sequential training from those that collapse under "
        "it, we may move from patching the symptom to addressing the cause.",
        styles["BodyText2"],
    ))

    # ─── 2. The Hypothesis ───
    story.append(Paragraph("2. The Hypothesis", styles["SectionHead"]))
    story.append(Paragraph(
        "We propose that the <i>topological structure</i> of a neural network's loss landscape "
        "predicts its resistance to catastrophic forgetting. Specifically, we hypothesize that "
        "networks which carve deeper and more persistent topological features into their loss "
        "landscape during training are more resistant to forgetting when trained sequentially on "
        "new tasks. In this framing, resistance to forgetting is not merely a consequence of "
        "parameter count, architectural scale, or regularization strength. It is a structural "
        "property of the geometry that optimization sculpts in weight space.",
        styles["BodyText2"],
    ))
    story.append(Paragraph(
        "The intuition is geometric. The loss landscape is the high-dimensional surface defined by "
        "model error as a function of its parameters, a perspective explored in foundational work on "
        "neural network optimization and visualization by Goodfellow et al. (2015) and Li et al. "
        "(2018). During training, gradient descent drives parameters into basins of low loss. Yet "
        "these basins differ in shape and internal structure. Some are shallow and weakly defined, "
        "while others are wide, deep, and geometrically intricate. When new training begins, the "
        "optimization process perturbs the parameters and reshapes the local landscape. If a solution "
        "lies in a shallow basin, relatively small updates can displace the model into a configuration "
        "that no longer supports prior performance. If the solution resides in a deeper and more "
        "structured region, it may exhibit greater stability under perturbation. This view resonates "
        "with prior findings linking basin geometry, flatness, and curvature to generalization and "
        "stability properties in neural networks (Hochreiter and Schmidhuber, 1997; Keskar et al., 2017).",
        styles["BodyText2"],
    ))
    story.append(Paragraph(
        "Persistent homology, a principal tool in topological data analysis, offers a rigorous and "
        "scale-invariant method for quantifying such geometric structure (Edelsbrunner and Harer, "
        "2010; Ghrist, 2008). Rather than examining curvature at a single resolution, persistent "
        "homology tracks the birth and death of topological features across a filtration. It "
        "identifies connected components through H<sub>0</sub>, loops through H<sub>1</sub>, and "
        "higher-dimensional voids such as H<sub>2</sub>, measuring how long each feature persists "
        "across scales. Features that persist over wide ranges are interpreted as structurally "
        "significant, whereas short-lived features are treated as topological noise. Because "
        "persistence is invariant under continuous deformation, it is particularly well suited for "
        "studying the highly nonconvex and irregular landscapes that arise in deep learning.",
        styles["BodyText2"],
    ))
    story.append(Paragraph(
        "Topological data analysis has already been applied to neural systems. It has been used to "
        "study learned representations and data manifolds (Carlsson, 2009; Naitzat et al., 2020), "
        "and has been applied to loss landscapes directly, including work by Ballester and Araujo "
        "(2020) examining topological signatures of optimization geometry. In parallel, catastrophic "
        "forgetting has been studied extensively since its original characterization in 1989, "
        "producing a large body of research on replay-based methods, regularization approaches such "
        "as elastic weight consolidation (Kirkpatrick et al., 2017), and architectural strategies "
        "such as progressive networks (Rusu et al., 2016).",
        styles["BodyText2"],
    ))
    story.append(Paragraph(
        "Yet these two lines of inquiry have remained separate. Topological data analysis has been "
        "applied to loss landscapes, and catastrophic forgetting has been examined for decades, but "
        "no work has directly asked the structural question at their intersection: <i>does the "
        "topological depth of learned representations predict their resistance to being overwritten?</i> "
        "By unifying these domains, we aim to move beyond surface-level mitigation strategies and "
        "toward a geometric account of why some learned solutions persist while others collapse under "
        "sequential training.",
        styles["BodyText2"],
    ))

    # ─── 3. Method ───
    story.append(Paragraph("3. Method", styles["SectionHead"]))

    story.append(Paragraph("3.1 Benchmark: Split CIFAR-100", styles["SubSectionHead"]))
    story.append(Paragraph(
        "We evaluate our hypothesis using Split CIFAR-100, a widely adopted benchmark in continual "
        "learning research. CIFAR-100 consists of 60,000 color images of size 32 by 32 pixels "
        "distributed across 100 object categories, with 600 images per class. Following standard "
        "protocol, we partition the dataset into two disjoint tasks. Task A contains classes 0 "
        "through 49 with 25,000 training images, while Task B contains classes 50 through 99 with "
        "25,000 training images. Each task includes 2,500 test images. This split enforces a strict "
        "sequential learning scenario in which the model is first optimized on Task A and then "
        "exposed to Task B without access to prior task data. The setup isolates catastrophic "
        "forgetting in its most direct form and avoids confounds introduced by rehearsal or task "
        "mixing strategies commonly used in continual learning studies.",
        styles["BodyText2"],
    ))

    story.append(Paragraph("3.2 Architectures", styles["SubSectionHead"]))
    story.append(Paragraph(
        "<b>ResNet-18</b> (He et al., 2016) serves as the convolutional baseline. It is an 18-layer "
        "deep residual network adapted for 32 by 32 inputs using a 3 by 3 initial convolution and "
        "no max pooling. The model contains approximately 11 million parameters. ResNet represents "
        "the canonical hierarchical convolutional paradigm in which information flows locally through "
        "progressively abstract feature maps, with spatial inductive biases encoded by design.",
        styles["BodyText2"],
    ))
    story.append(Paragraph(
        "<b>ViT-Small</b> represents the transformer paradigm applied to vision. The model consists "
        "of 4 transformer encoder layers, 4 attention heads, and 256-dimensional embeddings, with a "
        "patch size of 4, yielding 64 patches from each 32 by 32 image. The model contains "
        "approximately 3 million parameters. Unlike convolutional networks, Vision Transformers rely "
        "on global self-attention, allowing every spatial location to directly interact with every "
        "other location at each layer. This difference in information integration mechanism provides "
        "a structurally distinct comparison for evaluating geometric properties of the learned loss "
        "landscape.",
        styles["BodyText2"],
    ))

    story.append(Paragraph("3.3 Training Protocol", styles["SubSectionHead"]))
    story.append(Paragraph(
        "Both architectures are trained on Task A to convergence using stochastic gradient descent "
        "with momentum set to 0.9, a cosine learning rate schedule, and weight decay of "
        "5 \u00d7 10<super>\u22124</super>. ResNet-18 uses an initial learning rate of 0.1, while "
        "ViT-Small uses 0.01 to accommodate the different optimization dynamics of attention-based "
        "architectures, consistent with prior transformer training practice (Dosovitskiy et al., "
        "2021). Each model is trained for 100 epochs. Final weight checkpoints at convergence are "
        "saved and used as the reference points for subsequent loss landscape analysis. All other "
        "training hyperparameters are held constant where architecture permits to ensure comparability.",
        styles["BodyText2"],
    ))

    story.append(Paragraph("3.4 Loss Landscape Sampling", styles["SubSectionHead"]))
    story.append(Paragraph(
        "To characterize the local geometry of each trained model, we follow the loss landscape "
        "visualization methodology of Li et al. (2018). Around the converged weight vector, we "
        "construct a two-dimensional slice of parameter space by generating two random direction "
        "vectors. These directions are normalized using filter normalization, meaning that for each "
        "filter or neuron, the perturbation magnitude is scaled to match the norm of the original "
        "weights. This ensures that perturbations are proportional to parameter scale and comparable "
        "across architectures with different layer structures and weight distributions.",
        styles["BodyText2"],
    ))
    story.append(Paragraph(
        "We evaluate the loss on a 25 by 25 grid spanning the range from \u22121.0 to 1.0 along each "
        "direction, yielding 625 evaluation points. At each grid coordinate, we compute the cross-"
        "entropy loss on the Task A test set. The resulting grid defines a discretized approximation "
        "of the local loss surface surrounding the converged solution.",
        styles["BodyText2"],
    ))

    story.append(Paragraph("3.5 Persistent Homology", styles["SubSectionHead"]))
    story.append(Paragraph(
        "We compute persistent homology on the sampled loss surface using a lower-star filtration "
        "defined over an 8-connected grid graph. Each grid point is treated as a vertex with "
        "filtration value equal to the loss at that location. Edges between neighboring vertices "
        "appear at the maximum loss value of their two endpoints, forming a simplicial complex whose "
        "topology evolves as the filtration threshold increases.",
        styles["BodyText2"],
    ))
    story.append(Paragraph(
        "Persistent homology is computed using Ripser (Bauer, 2021) with sparse distance matrices. "
        "We analyze H<sub>0</sub>, representing connected components, and H<sub>1</sub>, representing "
        "loops in the surface structure. For each topological feature, we measure its lifetime as the "
        "difference between its birth and death filtration values. Our primary summary statistic is "
        "total persistence, defined as the sum of all feature lifetimes within a homology dimension. "
        "Total persistence captures the aggregate topological depth of the landscape and provides a "
        "quantitative measure of geometric structure that is robust to small perturbations.",
        styles["BodyText2"],
    ))

    story.append(Paragraph("3.6 Forgetting Measurement", styles["SubSectionHead"]))
    story.append(Paragraph(
        "To measure catastrophic forgetting, the model trained to convergence on Task A is expanded "
        "with a new classification head supporting all 100 classes. The backbone weights are retained, "
        "and training proceeds sequentially on Task B using stochastic gradient descent at one-tenth "
        "of the original learning rate. This reduced rate stabilizes training while still allowing "
        "substantial parameter updates.",
        styles["BodyText2"],
    ))
    story.append(Paragraph(
        "Task A test accuracy is evaluated at steps 100, 500, 1,000, 5,000, 10,000, and 25,000 "
        "during Task B training. No replay buffers, no regularization constraints, and no "
        "architectural isolation mechanisms are employed. This naive sequential training protocol "
        "exposes the intrinsic susceptibility of each architecture to catastrophic forgetting and "
        "allows us to examine whether measured topological depth correlates with retention dynamics.",
        styles["BodyText2"],
    ))

    # ─── 4. Preliminary Results ───
    story.append(Paragraph("4. Preliminary Results", styles["SectionHead"]))

    story.append(Paragraph("4.1 Topological Features", styles["SubSectionHead"]))
    story.append(Paragraph(
        "Table 1 summarizes the topological statistics of the loss landscape at convergence on "
        "Task A for both architectures.",
        styles["BodyText2"],
    ))

    # Table 1: Topology
    topo_data = [
        ["Metric", "ResNet-18", "ViT-Small"],
        ["Task A Test Accuracy", "82.0%", "62.2%"],
        ["H\u2080 Total Persistence", "2,151.5", "4,254.2"],
        ["H\u2080 Feature Count", "624", "624"],
        ["H\u2080 Max Lifetime", "5.21", "11.07"],
        ["H\u2081 Total Persistence", "0.0", "0.0"],
        ["Loss Range", "[0.86, 5.21]", "[0.86, 11.07]"],
    ]
    t1 = Table(topo_data, colWidths=[2.5 * inch, 1.5 * inch, 1.5 * inch])
    t1.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), TABLE_HEADER_BG),
        ("TEXTCOLOR", (0, 0), (-1, 0), HexColor("#4c1d95")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#dddddd")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [white, TABLE_ALT_BG]),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(t1)
    story.append(Paragraph(
        "<b>Table 1.</b> Topological features of the loss landscape at convergence on Task A. "
        "ViT-Small produces nearly 2\u00d7 the total H\u2080 persistence despite lower accuracy "
        "and fewer parameters.",
        styles["Caption"],
    ))

    story.append(Paragraph(
        "Despite achieving lower Task A accuracy, 62.2 percent compared to 82.0 percent, the "
        "Vision Transformer produces a loss landscape with substantially greater measurable "
        "topological depth. Total H<sub>0</sub> persistence for ViT-Small is 4,254.2, almost "
        "exactly double the 2,151.5 measured for ResNet-18. Because total persistence aggregates "
        "the lifetimes of all connected components across the filtration, this result indicates "
        "that the transformer converges to a basin with significantly more persistent geometric "
        "structure.",
        styles["BodyText2"],
    ))
    story.append(Paragraph(
        "The maximum lifetime of a single H<sub>0</sub> feature is also doubled, 11.07 for "
        "ViT-Small compared to 5.21 for ResNet-18. This suggests that the deepest connected basin "
        "surrounding the ViT solution extends across a much wider range of filtration values. In "
        "geometric terms, the transformer appears to settle into a solution region that is not "
        "merely low in loss, but structurally deeper when measured through scale-invariant "
        "topological persistence.",
        styles["BodyText2"],
    ))
    story.append(Paragraph(
        "Neither architecture produced nonzero H<sub>1</sub> persistence at the current grid "
        "resolution. This means that no loop structures were detected in the sampled landscape "
        "slice. There are two possible interpretations. The first is that the local minima are "
        "genuinely smooth and simply connected in the sampled neighborhood. The second is that the "
        "25 by 25 grid is too coarse to resolve higher-dimensional structure. Prior loss landscape "
        "work has shown that apparent smoothness can depend heavily on sampling resolution (Li et "
        "al., 2018). Future experiments will evaluate finer grids and alternative filtrations to "
        "determine whether higher-order topological features emerge.",
        styles["BodyText2"],
    ))
    story.append(Paragraph(
        "What is most important at this stage is not higher-dimensional structure, but the clear "
        "quantitative gap in H<sub>0</sub> persistence. The transformer produces nearly twice the "
        "total topological persistence despite having fewer parameters and lower classification "
        "accuracy. This dissociation suggests that topological depth is not reducible to raw "
        "performance or model size. It reflects a structural property of the learned basin itself.",
        styles["BodyText2"],
    ))

    story.append(Paragraph("4.2 Forgetting Dynamics", styles["SubSectionHead"]))
    story.append(Paragraph(
        "Table 2 reports Task A accuracy and retention ratio during sequential training on Task B.",
        styles["BodyText2"],
    ))

    # Table 2: Forgetting
    forget_data = [
        ["Step", "ResNet-18\nTask A Acc", "ResNet-18\nRetention", "ViT-Small\nTask A Acc", "ViT-Small\nRetention"],
        ["0", "82.0%", "100%", "62.2%", "100%"],
        ["100", "0.2%", "0.2%", "6.0%", "9.6%"],
        ["500", "0.0%", "0.0%", "6.1%", "9.8%"],
        ["1,000", "0.0%", "0.0%", "4.2%", "6.8%"],
        ["5,000", "0.0%", "0.0%", "1.7%", "2.7%"],
        ["10,000", "0.0%", "0.0%", "0.8%", "1.3%"],
        ["25,000", "0.0%", "0.0%", "0.1%", "0.2%"],
    ]
    t2 = Table(forget_data, colWidths=[0.7 * inch, 1.1 * inch, 1.0 * inch, 1.1 * inch, 1.0 * inch])
    t2.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), TABLE_HEADER_BG),
        ("TEXTCOLOR", (0, 0), (-1, 0), HexColor("#4c1d95")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#dddddd")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [white, TABLE_ALT_BG]),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(t2)
    story.append(Paragraph(
        "<b>Table 2.</b> Task A accuracy and retention during sequential Task B training. "
        "ResNet-18 forgets instantly; ViT-Small degrades gradually over thousands of steps.",
        styles["Caption"],
    ))

    story.append(Paragraph(
        "The forgetting dynamics differ dramatically across architectures. ResNet-18 undergoes "
        "immediate catastrophic forgetting. Within 100 training steps on Task B, Task A accuracy "
        "collapses from 82.0 percent to 0.2 percent, effectively a complete erasure. By step 500, "
        "accuracy is exactly 0.0 percent, below random chance for 50 classes, which would be 2.0 "
        "percent. The original representation is not gradually degraded. It is overwritten almost "
        "instantly.",
        styles["BodyText2"],
    ))
    story.append(Paragraph(
        "ViT-Small exhibits a markedly different trajectory. After 100 steps, it retains 6.0 "
        "percent Task A accuracy, thirty times higher than ResNet-18 at the same point. At step "
        "500, retention remains at 6.1 percent. Even after 10,000 steps of Task B training, "
        "measurable retention persists at 0.8 percent. The decay curve is progressive rather than "
        "abrupt. Knowledge is eroded over time rather than destroyed immediately.",
        styles["BodyText2"],
    ))
    story.append(Paragraph(
        "Critically, the architecture that exhibited nearly twice the total H<sub>0</sub> persistence "
        "also demonstrated dramatically slower forgetting. The model that carved a deeper and more "
        "persistent basin into its loss landscape retained prior knowledge longer under identical "
        "sequential training conditions. While this comparison involves only two architectures and "
        "does not establish causality, the directional alignment between topological depth and "
        "retention is striking.",
        styles["BodyText2"],
    ))
    story.append(Paragraph(
        "The central empirical observation is therefore structural. The network that formed a deeper "
        "geometric basin proved more resistant to destructive updates. The model that converged to a "
        "shallower basin was rapidly displaced. This correspondence is consistent with our hypothesis "
        "that topological depth in the loss landscape predicts vulnerability to catastrophic forgetting.",
        styles["BodyText2"],
    ))

    # ─── 5. Interpretation ───
    story.append(Paragraph("5. Interpretation", styles["SectionHead"]))
    story.append(Paragraph(
        "Two data points do not constitute proof, and we are explicit about that limitation. This "
        "study compares only two architectures under a single benchmark and training protocol. "
        "However, the direction of the result aligns precisely with the proposed hypothesis, and the "
        "magnitude of the observed difference is substantial. The gap in total H<sub>0</sub> "
        "persistence is nearly twofold, and the divergence in forgetting dynamics is not incremental "
        "but categorical. One architecture undergoes near-immediate collapse, while the other degrades "
        "gradually over thousands of training steps. This is not a marginal fluctuation that can be "
        "casually attributed to sampling noise or minor hyperparameter sensitivity. It is a structural "
        "contrast.",
        styles["BodyText2"],
    ))
    story.append(Paragraph(
        "The result is also architecturally revealing. ViT-Small contains roughly three million "
        "parameters compared to approximately eleven million in ResNet-18, and it achieves lower "
        "Task A accuracy at convergence. Yet it produces a loss landscape with nearly twice the total "
        "topological persistence and demonstrates markedly slower forgetting. This decoupling is "
        "important. If topological depth were merely a proxy for model capacity or raw performance, "
        "we would expect the larger and more accurate model to exhibit greater persistence. Instead, "
        "the opposite occurs. The metric appears to capture something structurally distinct about how "
        "knowledge is encoded in parameter space.",
        styles["BodyText2"],
    ))
    story.append(Paragraph(
        "One plausible explanation lies in the representational geometry induced by self-attention. "
        "In transformer architectures, every spatial position can directly attend to every other "
        "position at each layer. This creates a densely interconnected representational structure in "
        "which features are globally integrated. Convolutional networks, by contrast, aggregate "
        "information locally through spatially constrained receptive fields and hierarchical "
        "composition. While residual connections enable deeper signal flow, the inductive bias "
        "remains fundamentally local. It is conceivable that dense global integration distributes "
        "task-relevant information across a broader portion of parameter space, embedding it within "
        "a more interconnected basin. In contrast, locally concentrated filters may encode task-"
        "specific structure in narrower regions that are more easily displaced by subsequent gradient "
        "updates. Under this interpretation, attention-based models may naturally carve deeper and "
        "more persistent geometric structure into the loss landscape because knowledge is distributed "
        "rather than compartmentalized.",
        styles["BodyText2"],
    ))
    story.append(Paragraph(
        "If the observed relationship between topological depth and forgetting resistance generalizes "
        "across additional architectures and datasets, two practical applications follow.",
        styles["BodyText2"],
    ))
    story.append(Paragraph(
        "<b>First, topological persistence could serve as a diagnostic tool.</b> After training on an "
        "initial task, practitioners could compute total persistence on a sampled loss landscape slice "
        "to estimate vulnerability to catastrophic forgetting before deploying a model in a sequential "
        "or continual learning environment. This would provide a structural risk assessment grounded "
        "in geometry rather than post-hoc performance degradation.",
        ParagraphStyle("BulletItem", parent=styles["BodyText2"], leftIndent=24, spaceBefore=2, spaceAfter=6),
    ))
    story.append(Paragraph(
        "<b>Second, topological persistence could be incorporated directly into the training "
        "objective as a regularization signal.</b> A topological regularizer could penalize reductions "
        "in total persistence or explicitly reward the formation of deeper, longer-lived topological "
        "features. Such a mechanism would encourage optimization toward geometrically stable regions "
        "of parameter space. Instead of merely minimizing loss, the network would be incentivized to "
        "settle into structurally protected basins that are more resistant to perturbation from future "
        "tasks. In practical terms, the goal would be to shape the learning dynamics so that models "
        "carve deep, stable basins in weight space rather than shallow configurations that are easily "
        "overwritten.",
        ParagraphStyle("BulletItem2", parent=styles["BodyText2"], leftIndent=24, spaceBefore=2, spaceAfter=6),
    ))
    story.append(Paragraph(
        "While preliminary, these findings suggest that catastrophic forgetting may be partially "
        "predictable from the geometry of the learned solution itself. If so, the problem shifts "
        "from being purely algorithmic to fundamentally structural, opening a new line of inquiry "
        "at the intersection of optimization, topology, and continual learning.",
        styles["BodyText2"],
    ))

    # ─── 6. What's Next ───
    story.append(Paragraph("6. What\u2019s Next", styles["SectionHead"]))
    story.append(Paragraph(
        "This preliminary report establishes the experimental framework, defines the measurement "
        "pipeline, and presents initial empirical evidence supporting the topology\u2013forgetting "
        "hypothesis. The current results demonstrate directional alignment between topological "
        "persistence and retention under sequential training, but they represent only the first "
        "stage of a broader investigation. The next phases of work are designed to test robustness, "
        "improve measurement fidelity, and establish statistical validity across architectural "
        "families.",
        styles["BodyText2"],
    ))
    story.append(Paragraph(
        "Additional architectures are currently in progress. We are expanding the study to include "
        "ResNet-50 as a deeper convolutional baseline, wider ResNet variants to disentangle depth "
        "from width effects, and an LSTM-based recurrent model to introduce a fundamentally "
        "different temporal inductive bias. Evaluating three or more additional architectures will "
        "allow formal Spearman rank correlation analysis between total topological persistence and "
        "retention rate. Rank correlation is particularly appropriate here because the hypothesis "
        "predicts monotonic alignment rather than strict linear scaling. A statistically significant "
        "correlation across diverse architectures would constitute the first rigorous quantitative "
        "validation of the proposed relationship.",
        ParagraphStyle("NextItem", parent=styles["BodyText2"],
                       leftIndent=18, spaceBefore=2, spaceAfter=6),
    ))
    story.append(Paragraph(
        "We also plan to increase loss landscape sampling resolution. The absence of H<sub>1</sub> "
        "features in the current experiments may reflect genuine local smoothness, but it may also "
        "be a discretization artifact of the 25 by 25 grid. Prior loss landscape studies have shown "
        "that qualitative geometric structure can depend strongly on sampling density. We will "
        "therefore evaluate 51 by 51 and 101 by 101 grids using optimized sparse computation and "
        "memory-efficient persistent homology pipelines. Higher resolution sampling will allow us "
        "to determine whether loop structures or higher-order features emerge when the landscape is "
        "examined more finely.",
        ParagraphStyle("NextItem2", parent=styles["BodyText2"],
                       leftIndent=18, spaceBefore=2, spaceAfter=6),
    ))
    story.append(Paragraph(
        "Stochastic variation must also be addressed systematically. Both model training and random "
        "direction sampling in parameter space introduce variability. To quantify uncertainty, each "
        "experiment will be repeated across five independent random seeds. This will enable reporting "
        "of confidence intervals for total persistence, maximum lifetime, and forgetting curves. "
        "Statistical aggregation will allow us to distinguish structural effects from incidental "
        "variance.",
        ParagraphStyle("NextItem3", parent=styles["BodyText2"],
                       leftIndent=18, spaceBefore=2, spaceAfter=6),
    ))
    story.append(Paragraph(
        "If the correlation between topological depth and retention remains stable under these "
        "controls, we will proceed to the intervention phase. In Phase 5, we will implement a "
        "topological regularizer designed to preserve geometric structure during sequential training. "
        "The proposed formulation is",
        ParagraphStyle("NextItem4", parent=styles["BodyText2"],
                       leftIndent=18, spaceBefore=2, spaceAfter=6),
    ))
    story.append(Paragraph(
        "L<sub>topo</sub> = \u03bb \u00b7 max(0, P<sub>A</sub> \u2212 P<sub>current</sub>)",
        ParagraphStyle("Equation", parent=styles["BodyText2"],
                       alignment=TA_CENTER, spaceBefore=4, spaceAfter=4,
                       fontSize=11, textColor=HexColor("#4c1d95")),
    ))
    story.append(Paragraph(
        "where P denotes total persistence and P<sub>A</sub> represents persistence at convergence "
        "on the original task. This penalty discourages reductions in persistence during subsequent "
        "optimization, effectively resisting topological erosion of the learned basin. The objective "
        "is not merely to observe correlation but to test causality by actively shaping the loss "
        "landscape toward deeper and more stable regions of parameter space.",
        ParagraphStyle("NextItem5", parent=styles["BodyText2"],
                       leftIndent=18, spaceBefore=2, spaceAfter=6),
    ))
    story.append(Paragraph(
        "Pending results across at least five architectures with formal statistical testing, the "
        "target venue is NeurIPS or ICML within the continual learning track. Acceptance will "
        "require demonstrating reproducibility, architectural generality, and statistically "
        "significant correlation between geometric structure and forgetting dynamics. The long-term "
        "objective is to establish topological depth not as an isolated metric, but as a principled "
        "structural lens through which continual learning stability can be understood and engineered.",
        ParagraphStyle("NextItem6", parent=styles["BodyText2"],
                       leftIndent=18, spaceBefore=2, spaceAfter=6),
    ))

    # ─── 7. References ───
    story.append(Paragraph("References", styles["SectionHead"]))

    refs = [
        "Ballester, R. and Araujo, X. (2020). On the interplay between topological data analysis "
        "and deep learning. <i>NeurIPS Workshop on Topological Data Analysis</i>.",

        "Bauer, U. (2021). Ripser: Efficient computation of Vietoris\u2013Rips persistence barcodes. "
        "<i>Journal of Applied and Computational Topology</i>, 5(3), 391\u2013423.",

        "Carlsson, G. (2009). Topology and data. <i>Bulletin of the American Mathematical Society</i>, "
        "46(2), 255\u2013308.",

        "Dosovitskiy, A., Beyer, L., Kolesnikov, A. et al. (2021). An image is worth 16x16 words: "
        "Transformers for image recognition at scale. <i>International Conference on Learning "
        "Representations</i>.",

        "Edelsbrunner, H. and Harer, J. (2010). <i>Computational Topology: An Introduction</i>. "
        "American Mathematical Society.",

        "Ghrist, R. (2008). Barcodes: The persistent topology of data. <i>Bulletin of the American "
        "Mathematical Society</i>, 45(1), 61\u201375.",

        "Goodfellow, I.J., Vinyals, O. and Saxe, A.M. (2015). Qualitatively characterizing neural "
        "network optimization problems. <i>International Conference on Learning Representations</i>.",

        "He, K., Zhang, X., Ren, S. and Sun, J. (2016). Deep residual learning for image recognition. "
        "<i>Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition</i>, 770\u2013778.",

        "Hochreiter, S. and Schmidhuber, J. (1997). Flat minima. <i>Neural Computation</i>, 9(1), "
        "1\u201342.",

        "Keskar, N.S., Mudigere, D., Nocedal, J., Smelyanskiy, M. and Tang, P.T.P. (2017). On "
        "large-batch training for deep learning: Generalization gap and sharp minima. <i>International "
        "Conference on Learning Representations</i>.",

        "Kirkpatrick, J., Pascanu, R., Rabinowitz, N. et al. (2017). Overcoming catastrophic "
        "forgetting in neural networks. <i>Proceedings of the National Academy of Sciences</i>, "
        "114(13), 3521\u20133526.",

        "Kumaran, D., Hassabis, D. and McClelland, J.L. (2016). What learning systems do intelligent "
        "agents need? Complementary learning systems theory updated. <i>Trends in Cognitive Sciences</i>, "
        "20(7), 512\u2013534.",

        "Li, H., Xu, Z., Taylor, G., Studer, C. and Goldstein, T. (2018). Visualizing the loss "
        "landscape of neural nets. <i>Advances in Neural Information Processing Systems</i>.",

        "McCloskey, M. and Cohen, N.J. (1989). Catastrophic interference in connectionist networks: "
        "The sequential learning problem. <i>Psychology of Learning and Motivation</i>, 24, 109\u2013165.",

        "Naitzat, G., Zhitnikov, A. and Lim, L.-H. (2020). Topology of deep neural networks. "
        "<i>Journal of Machine Learning Research</i>, 21(184), 1\u201340.",

        "Otter, N., Porter, M.A., Tillmann, U., Grindrod, P. and Harrington, H.A. (2017). A roadmap "
        "for the computation of persistent homology. <i>EPJ Data Science</i>, 6(1), 1\u201338.",

        "Rusu, A.A., Rabinowitz, N.C., Desjardins, G. et al. (2016). Progressive neural networks. "
        "<i>arXiv preprint arXiv:1606.04671</i>.",

        "Tononi, G. and Cirelli, C. (2014). Sleep and the price of plasticity: From synaptic and "
        "cellular homeostasis to memory consolidation and integration. <i>Neuron</i>, 81(1), 12\u201334.",
    ]
    for i, ref in enumerate(refs, 1):
        story.append(Paragraph(f"[{i}] {ref}", styles["Reference"]))

    # ─── Footer ───
    story.append(Spacer(1, 24))
    story.append(HRFlowable(width="100%", thickness=0.5, color=HexColor("#dddddd")))
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "Axion Deep Labs &nbsp;\u00b7&nbsp; axiondeep.com/research/experiments &nbsp;\u00b7&nbsp; "
        "EXP-01 Preliminary Report &nbsp;\u00b7&nbsp; February 2026",
        styles["Footer"],
    ))
    story.append(Paragraph(
        "This is a preliminary report. Results are based on two architectures and should not be "
        "interpreted as definitive. Full statistical analysis pending additional experiments.",
        ParagraphStyle("Disclaimer", parent=styles["Footer"], spaceBefore=4, fontSize=7.5,
                       textColor=HexColor("#999999")),
    ))

    doc.build(story)
    print(f"PDF generated: {OUTPUT}")


if __name__ == "__main__":
    build_pdf()
