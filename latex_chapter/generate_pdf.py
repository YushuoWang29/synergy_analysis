#!/usr/bin/env python3
"""
Generate a PDF document for the PCA-based hand synergy analysis chapter.
Uses fpdf2 for PDF generation with embedded figures.
"""
import os, sys
from fpdf import FPDF

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(SCRIPT_DIR, 'figures')
OUTPUT_PATH = os.path.join(SCRIPT_DIR, 'chapter_synergy_pca.pdf')

class ChapterPDF(FPDF):
    def __init__(self):
        super().__init__('P', 'mm', 'A4')
        # Register Chinese font - use SimHei or Microsoft YaHei
        font_paths = [
            r'C:\Windows\Fonts\msyh.ttc',      # Microsoft YaHei
            r'C:\Windows\Fonts\simhei.ttf',     # SimHei
            r'C:\Windows\Fonts\msyhbd.ttc',     # Microsoft YaHei Bold
        ]
        self.cn_font = None
        for fp in font_paths:
            if os.path.exists(fp):
                self.add_font('CN', '', fp, uni=True)
                # Try bold variant
                bfp = fp.replace('.ttc', 'bd.ttc').replace('.ttf', 'bd.ttf')
                if os.path.exists(bfp):
                    self.add_font('CN', 'B', bfp, uni=True)
                else:
                    self.add_font('CN', 'B', fp, uni=True)
                self.cn_font = 'CN'
                break
        
        # Fallback to DejaVu if no Chinese font
        if not self.cn_font:
            self.add_font('DejaVu', '', r'C:\Windows\Fonts\DejaVuSans.ttf', uni=True)
            self.add_font('DejaVu', 'B', r'C:\Windows\Fonts\DejaVuSans-Bold.ttf', uni=True)
            self.cn_font = 'DejaVu'

    def header(self):
        if self.page_no() > 1:
            self.set_font(self.cn_font, '', 8)
            self.set_text_color(128,128,128)
            self.cell(0, 5, '第{}章  基于主成分分析的手部运动协同特征提取'.format(
                self.page_no()), 0, 1, 'R')
            self.line(10, 12, 200, 12)
            self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font(self.cn_font, '', 8)
        self.set_text_color(128,128,128)
        self.cell(0, 10, f'- {self.page_no()} -', 0, 0, 'C')

    def chapter_title(self, title, level=0):
        """Add a chapter or section title."""
        if level == 0:
            self.set_font(self.cn_font, 'B', 18)
            self.ln(5)
            self.set_text_color(0, 51, 102)
            self.multi_cell(0, 10, title)
            self.ln(3)
        elif level == 1:
            self.set_font(self.cn_font, 'B', 14)
            self.ln(3)
            self.set_text_color(0, 80, 140)
            self.multi_cell(0, 8, title)
            self.ln(2)
        elif level == 2:
            self.set_font(self.cn_font, 'B', 11)
            self.ln(2)
            self.set_text_color(60, 60, 60)
            self.multi_cell(0, 7, title)
            self.ln(1)

    def body_text(self, text):
        """Add a paragraph of body text."""
        self.set_font(self.cn_font, '', 10)
        self.set_text_color(30, 30, 30)
        self.multi_cell(0, 5.5, text)
        self.ln(1)

    def body_bold(self, text):
        """Add bold inline text."""
        self.set_font(self.cn_font, 'B', 10)
        self.set_text_color(30, 30, 30)
        self.multi_cell(0, 5.5, text)
        self.ln(1)

    def add_image_centered(self, path, caption, w=160):
        """Add an image with caption."""
        if not os.path.exists(path):
            self.body_text(f'[图片未找到: {path}]')
            return
        self.ln(2)
        # Use image directly; fpdf2 auto-scales
        x = (210 - w) / 2
        self.image(path, x=x, w=w)
        self.ln(1)
        # Caption
        self.set_font(self.cn_font, '', 9)
        self.set_text_color(80, 80, 80)
        self.multi_cell(0, 4.5, caption)
        self.ln(3)

    def add_enumerated_list(self, items):
        """Add a numbered list."""
        self.set_font(self.cn_font, '', 10)
        self.set_text_color(30, 30, 30)
        for i, item in enumerate(items, 1):
            self.multi_cell(0, 5.5, f'  ({i}) {item}')
            self.ln(0.5)
        self.ln(1)

    def add_references(self, refs):
        """Add a reference list."""
        self.set_font(self.cn_font, '', 9)
        self.set_text_color(60, 60, 60)
        for ref in refs:
            self.multi_cell(0, 4.5, ref)
            self.ln(1)


def build_pdf():
    pdf = ChapterPDF()
    pdf.set_auto_page_break(auto=True, margin=20)
    
    # ---- Title page ----
    pdf.add_page()
    pdf.ln(40)
    pdf.set_font(pdf.cn_font, 'B', 22)
    pdf.set_text_color(0, 51, 102)
    pdf.multi_cell(0, 12, '第 X 章\n基于主成分分析的手部运动协同\n特征提取', align='C')
    pdf.ln(10)
    pdf.set_font(pdf.cn_font, '', 12)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 8, '—— 数据处理与分析章节 ——', 0, 1, 'C')
    pdf.ln(30)
    pdf.set_font(pdf.cn_font, '', 10)
    pdf.cell(0, 7, '数据来源：MyoKi 数据集 & Ninapro 数据库', 0, 1, 'C')
    pdf.cell(0, 7, '分析方法：主成分分析（PCA）', 0, 1, 'C')
    pdf.ln(30)
    pdf.set_font(pdf.cn_font, 'B', 10)
    pdf.set_text_color(0,0,0)
    pdf.cell(0, 7, '包含图1-图7 共7张分析图表', 0, 1, 'C')
    
    # ---- Chapter body ----
    pdf.add_page()
    pdf.chapter_title('第X章  基于主成分分析的手部运动协同特征提取', 0)
    
    # === Section 1: Introduction ===
    pdf.chapter_title('1  引言：高维手部控制的挑战', 1)
    
    pdf.body_text(
        '人手具有超过20个自由度（Degree of Freedom, DoF），能够执行从精密捏取到强力抓握的多种复杂操作。'
        '从运动控制的视角来看，中枢神经系统（Central Nervous System, CNS）需要同时协调大量的肌肉和关节来实现这些精细运动。'
        '然而，控制理论的研究表明，实时独立控制每个自由度在计算上是极其困难的，'
        '这种现象被称为"维度灾难"（Curse of Dimensionality）[1]。'
    )
    pdf.body_text(
        '神经生理学和生物力学研究发现，CNS并不独立控制每个关节，而是采用"运动协同"（Motor Synergy）的策略[2]，'
        '即通过一组固定的协同模式（Synergy Patterns）来简化控制。'
        'Santello等人[1]最早使用主成分分析（Principal Component Analysis, PCA）对人类抓取过程中的手部姿态进行研究，'
        '发现前两个主成分即可解释约80%的手部姿态方差。这一发现为仿人灵巧手的简化控制提供了重要的理论依据。'
    )
    pdf.body_text(
        '本章采用PCA方法，对多参与者手部运动数据集进行系统的协同分析。我们旨在回答以下关键问题：'
        '（1）人类手部运动的主要协同模式是什么？'
        '（2）这些协同模式的普遍性如何？'
        '（3）数据驱动的协同与基于理论的协同（如Pisa/IIT SoftHand）有何异同？'
    )
    
    # === Section 2: Data Sources ===
    pdf.chapter_title('2  数据来源', 1)
    
    pdf.chapter_title('2.1  MyoKi数据集', 2)
    pdf.body_text(
        '本研究主要采用MyoKi（MyoKinesthetic）数据集[3]作为主要分析对象。'
        'MyoKi数据集使用CyberGlove II数据手套，记录了12名健康参与者（编号P01-P12）'
        '在44种不同任务下的手部运动数据。每段动作被重采样至100帧，记录了18个通道的关节角度数据。'
    )
    pdf.body_text(
        '根据Task_categorization.xlsx中的任务分类，我们将44种任务分为抓取类任务'
        '（包括圆柱抓取、球状抓取、钩状抓取、三指捏取、精确捏取和片状抓取共6种类型）'
        '和其他非抓取任务。基于手部协同分析的研究目标，我们主要聚焦于抓取类任务的静态姿势分析。'
    )
    
    pdf.chapter_title('2.2  Ninapro数据集', 2)
    pdf.body_text(
        '为了验证协同模式的跨数据集普适性，我们同时采用了Ninapro数据库[4]中的三个子数据集'
        '（E1、E2、E3）进行对照分析。Ninapro数据使用多种数据手套记录，其预处理流程与MyoKi类似。'
    )
    
    # === Section 3: Preprocessing ===
    pdf.chapter_title('3  数据预处理', 1)
    pdf.body_text(
        '数据预处理是确保后续PCA分析可靠性的关键步骤，主要包括以下环节：'
    )
    
    items = [
        '数据清洗与通道选择：CyberGlove II数据手套提供18个传感器通道。根据解剖学分析，'
        '我们剔除了腕部屈伸/桡尺偏（2通道）和掌弓（1通道）通道，仅保留15个手部关节自由度。'
        '具体包括：拇指外展、拇指MCP、拇指IP、拇指对掌；各手指的MCP和PIP关节；'
        '以及指间外展/内收关节。这一选择与Santello（1998）的15自由度模型一致[1]。',
        
        '动作切分：利用实验任务标记信号（restimulus/restrial），将连续数据流切割为独立试次。',
        
        '运动/静态筛选：对每个试次计算关节角速度的模长，并设定阈值以区分动态运动阶段和静态保持阶段。'
        '本研究主要关注抓取动作末端的手部静态姿势（即参与者完成抓取后的稳定姿态），'
        '因为这最能反映不同抓取类型的差异。',
        
        '重采样：将每个试次统一重采样到100帧（TARGET_FRAMES=100），确保所有试次的时间维度一致。',
        
        '静态姿势提取：从每个试次的最后25%帧中取平均作为该试次的静态姿势表征。'
    ]
    pdf.add_enumerated_list(items)
    
    pdf.body_text(
        '经过上述预处理，我们从MyoKi数据集中提取了共4883个试次水平的静态姿势数据（全任务），'
        '其中1188个试次属于抓取类任务。'
    )
    
    # === Section 4: PCA Principles ===
    pdf.chapter_title('4  主成分分析原理', 1)
    pdf.body_text(
        '主成分分析是一种经典的无监督降维方法，其核心思想是通过线性变换将原始高维数据投影到方差最大的方向上。'
        '对于15维的手部关节角度数据 x in R^15，PCA寻找一组正交基向量 {w1, w2, ..., w15}，'
        '使得投影后的数据 yi = x^T wi 的方差最大化。'
    )
    pdf.body_text(
        '数学上，这等价于对数据协方差矩阵 Sigma = (1/(N-1)) * sum((xi - x_mean)(xi - x_mean)^T) '
        '进行特征值分解：Sigma * wi = lambda_i * wi，其中 lambda_i 为第i个特征值，'
        'wi 为对应的特征向量（即主成分负载向量）。'
        '第i个主成分解释的方差比率为 lambda_i / sum(lambda_j)。'
    )
    pdf.body_text(
        '在本研究中，主成分负载向量 wi 即为"特征姿势"（Eigenposture），'
        '反映了各关节在特定协同模式中的相对贡献和运动方向。'
        '每个抓取姿势可由主成分得分 y 的线性组合近似重建。'
    )
    
    # === Section 5: Results ===
    pdf.chapter_title('5  PCA分析结果', 1)
    
    # --- 5.1 Variance ---
    pdf.chapter_title('5.1  方差解释分析', 2)
    pdf.body_text(
        '我们对MyoKi数据集中全部1188个抓取类静态姿势执行PCA，得到15个主成分的方差解释率。'
        '图1(a)展示了各主成分的方差解释比率，图1(b)为累计方差解释曲线。'
    )
    
    pdf.add_image_centered(
        os.path.join(FIG_DIR, 'fig01_variance.png'),
        '图1: PCA方差解释分析。（a）各主成分的个体方差解释比率，前三个主成分用红色突出显示；'
        '（b）累计方差解释曲线及90%、95%阈值。数据来源：MyoKi数据集全部抓取类静态姿势（N=1188）。',
        w=165
    )
    
    pdf.body_bold(
        '分析结果显示，PC1解释了 50.24% 的总方差，PC2解释了 15.84%，PC3解释了 9.94%。'
        '前三个主成分合计解释 76.02% 的方差，前六个主成分即可解释 91.37% 的方差，超过常用的90%阈值。'
        '相比之下，达到95%的方差解释水平需要8个主成分。'
    )
    pdf.body_text(
        '这一结果与Santello等人[1]的经典结论一致，即手部抓取运动可以有效地用少量协同模式进行描述。'
        '值得注意的是，PC1（约50%）在总方差中占据主导地位，表明所有抓取任务共享了一个主要的协同模式。'
    )
    
    # --- 5.2 Eigenpostures ---
    pdf.chapter_title('5.2  特征姿势分析', 2)
    pdf.body_text(
        '图2展示了前三个主成分对应的特征姿势（Eigenpostures），即主成分负载向量的条形图。'
        '各关节按手指分组着色（拇指为红色，食指为蓝色，中指为绿色，无名指为橙色，小指为紫色）。'
    )
    
    pdf.add_image_centered(
        os.path.join(FIG_DIR, 'fig02_eigenpostures.png'),
        '图2: 前三主成分特征姿势（Eigenpostures）。纵轴为负载权重，正权重表示该关节在相应协同模式中趋向屈曲/内收方向。'
        '关节缩写：ThumbAbd=拇指外展，ThumbMCP=拇指掌指关节，ThumbIP=拇指指间关节，ThumbOpp=拇指对掌。',
        w=170
    )
    
    pdf.body_text(
        '对PC1负载向量的分析揭示了一个清晰的模式：除拇指外展（ThumbAbd）和指间外展关节'
        '（IdxMidAbd, MidRingAbd, RingLitAbd）为负值外，其他所有关节均为正权重。'
        '这意味着PC1实际上捕捉了"手指屈曲与拇指对掌"的协同运动——正方向对应于手指弯曲'
        '（握拳动作）和拇指内收，负方向对应于手指伸展和拇指外展。'
        '这与Santello等人的发现一致[1]，即PC1代表了手部"开-合"的基本协同模式。'
    )
    pdf.body_text(
        'PC2则表现出明显的手指分组模式：拇指和食指（精确抓取所需的手指）与无名指和小指呈相反方向的负载。'
        'PC2正方向对应于拇指对掌（ThumbOpp权重0.423）与食指、无名指的协同屈曲，同时伴随环小指外展。'
        '这捕捉了"精确捏取 vs. 力量抓取"的对比关系。'
    )
    pdf.body_text(
        'PC3进一步细化了拇指的独立运动模式，拇指对掌关节（权重0.537）和拇指外展（权重0.414）占据主导，'
        '同时伴随食指与中指的相对运动。这一模式反映了需要对拇指进行精密定位的精细操作类型。'
    )
    
    # --- 5.3 PC space ---
    pdf.chapter_title('5.3  协同空间分布', 2)
    pdf.body_text(
        '图3将所有抓取姿势样本投影到PC1-PC2构成的二维协同空间。'
        '数据点（N=1188）的颜色编码显示了密度的差异，红色轮廓线代表概率密度等值线。'
    )
    
    pdf.add_image_centered(
        os.path.join(FIG_DIR, 'fig03_pc1_pc2_scatter.png'),
        '图3: 全部抓取静态姿势在PC1-PC2平面的投影。每个蓝点代表一个试次的静态抓取姿势。'
        '红色星号标记示意性的极端姿势位置：全力抓取（PC1正方向）、捏取姿势（PC2正方向）、'
        '全伸展（PC1负方向）和侧捏姿势（PC2负方向）。',
        w=150
    )
    
    pdf.body_text(
        '从图3可以观察到以下重要特征：'
        '（1）数据点在PC1方向（开-合协同）分布范围最广，反映了不同抓取类型从全伸展到全力抓握的连续变化；'
        '（2）PC2方向上的分布揭示了任务相关的精细调节，精确抓取任务倾向于在PC2正方向区域；'
        '（3）不同任务在协同空间中形成聚类，但存在显著的类间重叠，表明自然抓取中存在连续而非离散的控制策略。'
    )
    
    # --- 5.4 Pisa comparison ---
    pdf.chapter_title('5.4  与Pisa/IIT理论协同的对比', 2)
    pdf.body_text(
        'Pisa/IIT SoftHand设计团队从解剖学和生物力学原理出发，提出了两种基于理论的手部协同模式：'
        '协同1（所有关节同向屈曲）和协同2（手指呈现梯度性屈曲）[5]。'
        '我们将数据驱动的PCA协同通过Procrustes旋转对齐后与理论协同进行比较，如图4所示。'
    )
    
    pdf.add_image_centered(
        os.path.join(FIG_DIR, 'fig04_pisa_rotation.png'),
        '图4: 数据驱动的PCA协同（原始PC列）与Pisa/IIT SoftHand理论协同的对比。'
        '通过Procrustes旋转将PCA协同旋转至与理论目标协同最近似的位置（Pisa旋转后列）。',
        w=170
    )
    
    pdf.body_text(
        '对比结果表明，经过Procrustes旋转后，PCA的第一协同与Pisa理论协同1（全手同向屈曲）高度吻合。'
        '旋转后的PC1在所有手指关节上呈现一致的正权重，与抓取动作的基本机制一致。'
        '第二协同虽然与理论协同2存在相似性（从食指到小指呈现梯度变化），'
        '但在拇指对掌和手指外展的权重分配上存在差异。'
        '这表明，尽管理论协同为灵巧手抓取提供了良好的基线，但真实人类手部的协同模式更为复杂和个性化。'
    )
    
    # --- 5.5 Participant comparison ---
    pdf.chapter_title('5.5  被试间对比分析', 2)
    pdf.body_text(
        '为评估PCA结果的个体间一致性，我们对各参与者的数据分别执行PCA，'
        '并将PC1、PC2的方差解释率进行跨被试对比（图5）。'
    )
    
    pdf.add_image_centered(
        os.path.join(FIG_DIR, 'fig05_participant_comparison.png'),
        '图5: 11名参与者的PCA方差解释对比。柱状图显示各被试的PC1（红色）和PC2（蓝色）方差解释率，'
        '绿色折线显示PC1+PC2的合计方差解释率。',
        w=160
    )
    
    pdf.body_text(
        '跨被试分析显示，PC1的方差解释率在不同个体间稳定保持在44%-55%之间，'
        'PC1+PC2合计在60%-72%之间。这种高度的一致性证实了基本手部协同模式的生物学普遍性，'
        '同时也提示个体差异（手部尺寸、柔韧性、抓取习惯等）对协同模式的精细结构产生影响。'
    )
    
    # --- 5.6 Task examples ---
    pdf.chapter_title('5.6  代表性抓取任务的手部姿态分析', 2)
    pdf.body_text(
        '为直观展示PCA协同对不同抓取类型的表征能力，我们选取了四种代表性抓取任务，'
        '分析它们在PC1-PC2空间中的位置和对应的关节角度配置（图7）。'
    )
    
    pdf.add_image_centered(
        os.path.join(FIG_DIR, 'fig07_task_examples.png'),
        '图6: 四种代表性抓取任务的静态手部姿态。任务编号和PC坐标（PC1, PC2）标注在子图标题中：'
        '（a）拿杯子——精确抓取；（b）拿球——球状抓取；（c）拿书——捏取；（d）提包——力量抓取。',
        w=160
    )
    
    pdf.body_text(
        '从图6可以清晰地看到不同类型抓取在关节空间中的差异化配置。'
        '精确抓取任务（拿杯子、拿书）表现出较明显的拇指对掌运动（ThumbOpp）和较小的手指屈曲角度，对应PC2正方向。'
        '而力量型抓取（提包）则表现为全手屈曲的协同模式，对应PC1正方向。'
    )
    
    # Add synergy concept figure
    pdf.chapter_title('5.7  协同分析原理示意图', 2)
    pdf.body_text(
        '为了直观地说明手部协同分析的核心概念——从高维手部姿态空间到低维协同空间的降维映射，'
        '图7展示了这一过程的示意图。'
    )
    
    pdf.add_image_centered(
        os.path.join(FIG_DIR, 'fig06_synergy_concept.png'),
        '图7: 手部姿态协同分析原理示意图。从左至右依次为：高维手部姿态空间（15自由度）、'
        'PCA降维映射、低维协同空间（3-6个主成分即可解释约90%方差）。',
        w=160
    )
    
    # === Section 6: Discussion ===
    pdf.chapter_title('6  讨论与结论', 1)
    
    pdf.body_bold('协同的层次结构')
    pdf.body_text(
        'PC1（约50%）代表了手部抓取中最重要的"开-合"协同模式，这一模式在所有抓取任务中普遍存在。'
        'PC2（约16%）区分了以拇指为主导的精确操作和以四指为主导的力量抓取。'
        'PC3（约10%）进一步细化了拇指的独立控制。'
        '这种从全局到局部的层次化组织，反映了CNS在运动控制中采用的分层策略。'
    )
    
    pdf.body_bold('协同的低维本质')
    pdf.body_text(
        '仅需3个协同即可解释超过75%的手部姿态方差，6个协同即可解释超过91%。'
        '这意味着一个15自由度的灵巧手系统，在95%的保真度下仅需约8个控制信号即可复现绝大多数的自然抓取姿态，'
        '这为低维度控制接口的设计提供了理论依据。'
    )
    
    pdf.body_bold('数据驱动 vs. 理论协同')
    pdf.body_text(
        '数据驱动的PCA协同与Pisa/IIT理论协同展现出良好的对应关系，特别是对于第一协同。'
        '但数据驱动方法揭示了个体差异和任务特异性，这为构建更灵活、更个性化的灵巧手控制策略提供了参考。'
    )
    
    pdf.body_bold('研究局限与展望')
    pdf.body_text(
        '本研究仅采用了抓取任务的静态姿势数据，未考虑动态运动过程中的协同变化。'
        '此外，PCA作为线性降维方法，在捕捉非线性协同关系方面存在局限。'
        '未来工作可以探索非线性降维方法（如自编码器、t-SNE等），'
        '以及考虑肌电信号（EMG）与运动学协同之间的映射关系。'
        '这些结论将为基于协同的仿人灵巧手控制策略设计提供重要的理论指导。'
    )
    
    # === References ===
    pdf.chapter_title('参考文献', 1)
    refs = [
        '[1] M. Santello, M. Flanders, and J. F. Soechting, "Postural hand synergies for tool use," '
        'Journal of Neuroscience, vol. 18, no. 23, pp. 10105-10115, 1998.',
        
        '[2] J. F. Soechting and M. Flanders, "Moving in three-dimensional space: frames of reference, '
        'vectors, and coordinate systems," Annual Review of Neuroscience, vol. 15, pp. 167-191, 1992.',
        
        '[3] V. S. F. et al., "MyoKi: A kinematic hand movement dataset," Scientific Data, 2023.',
        
        '[4] M. Atzori et al., "Electromyography data for non-invasive naturally-controlled robotic hand '
        'prostheses," Scientific Data, vol. 1, p. 140053, 2014.',
        
        '[5] A. Bicchi, M. Gabiccini, and M. Santello, "Modelling natural and artificial hands with '
        'synergies," Philosophical Transactions of the Royal Society B, vol. 366, no. 1581, pp. 3153-3161, 2011.',
        
        '[6] M. Santello and J. F. Soechting, "Gradual molding of the hand to object contours," '
        'Journal of Neurophysiology, vol. 79, no. 3, pp. 1307-1320, 1998.',
        
        '[7] A. d\'Avella, P. Saltiel, and E. Bizzi, "Combinations of muscle synergies in the construction '
        'of a natural motor behavior," Nature Neuroscience, vol. 6, no. 3, pp. 300-308, 2003.',
    ]
    pdf.add_references(refs)
    
    # Save
    pdf.output(OUTPUT_PATH)
    print(f'PDF saved to: {OUTPUT_PATH}')
    print(f'Pages: {pdf.page_no()}')
    return OUTPUT_PATH


if __name__ == '__main__':
    build_pdf()
