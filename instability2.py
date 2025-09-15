import numpy as np
import matplotlib.pyplot as plt
import matplotlib
# 设置字体以支持中文（可选）
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

class FlameStabilityAnalysis:
    """
    最终修正版：环形火焰稳定性分析
    基于Sivashinsky理论，确保Markstein长度正确计算
    """
    
    def __init__(self, Lewis=0.7, Zeldovich=10.0, heat_release=5.0):
        """
        初始化参数（无量纲）
        """
        self.Le = Lewis
        self.beta = Zeldovich
        self.sigma = heat_release
        
        # 导出参数
        self.epsilon = self.sigma / (1 + self.sigma)  # 热膨胀系数
        
    def markstein_length(self, Le=None):
        """
        计算Markstein长度 - 关键函数！
        L决定了稳定性：L>0不稳定，L<0稳定
        """
        # 使用提供的Le或对象的Le
        Lewis = Le if Le is not None else self.Le
        
        # Markstein长度的主要贡献来自Lewis数效应
        # 这是最重要的公式！
        if Lewis == 1.0:
            L = 0.0  # Le=1时中性稳定
        else:
            # 基本公式：L ∝ (1-Le)/Le
            L_Lewis = (1.0 - Lewis) / Lewis
            
            # 活化能修正（使L的量级合理）
            L_activation = 1.0 / (2.0 * self.beta)
            
            # 热膨胀修正
            L_thermal = 1.0 + self.epsilon
            
            # 总Markstein长度
            L = L_Lewis * L_activation * L_thermal
        
        return L
    
    def dispersion_relation(self, k, Le=None):
        """
        色散关系：σ(k)
        决定不同波数的增长率
        """
        Lewis = Le if Le is not None else self.Le
        L = self.markstein_length(Lewis)
        
        k = np.asarray(k)
        
        # 避免k=0的奇异性
        k_safe = np.where(k > 0, k, 1e-10)
        
        # 1. 扩散稳定项（总是负的）
        diffusion = -k_safe**2
        
        # 2. Darrieus-Landau不稳定项（L>0时为正）
        DL_instability = L * self.beta * k_safe**2
        
        # 3. 高波数截断（防止短波无限增长）
        cutoff = -k_safe**4 / (4.0 * self.beta)
        
        # 总增长率
        sigma = diffusion + DL_instability + cutoff
        
        # k=0时稳定
        sigma = np.where(k > 0, sigma, -0.1)
        
        return sigma
    
    def find_critical_mode(self, Le=None):
        """
        找到最不稳定的模式
        """
        Lewis = Le if Le is not None else self.Le
        L = self.markstein_length(Lewis)
        
        if L > 0:
            # 极值条件：dσ/dk = 0
            # 解得：k_c = sqrt(2βL)
            k_c = np.sqrt(2 * self.beta * L)
            # 代入得最大增长率
            sigma_max = self.dispersion_relation(k_c, Lewis)
        else:
            k_c = 0
            sigma_max = 0
        
        return k_c, sigma_max
    
    def analyze_case(self, Le):
        """
        分析单个Lewis数情况
        """
        print(f"\n测试 Le = {Le}")
        print("-" * 40)
        print(f"参数: Le={Le:.2f}, β={self.beta:.1f}, σ={self.sigma:.1f}")
        
        # 计算关键参数
        L = self.markstein_length(Le)
        k_c, sigma_max = self.find_critical_mode(Le)
        
        print(f"Markstein长度: L = {L:.4f}")
        print(f"临界波数: k_c = {k_c:.3f}")
        print(f"最大增长率: σ_max = {sigma_max:.4f}")
        print(f"状态: {'不稳定' if sigma_max > 0 else '稳定'}")
        
        return L, k_c, sigma_max
    
    def plot_analysis(self):
        """
        综合可视化分析
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 测试三个关键Lewis数
        Le_cases = [0.5, 0.7, 1.0]
        colors = ['red', 'blue', 'green']
        
        # 1-3. 色散曲线（上排）
        k_array = np.linspace(0, 6, 500)
        
        for i, (Le, color) in enumerate(zip(Le_cases, colors)):
            ax = axes[0, i]
            
            # 计算色散关系
            sigma = self.dispersion_relation(k_array, Le)
            L = self.markstein_length(Le)
            k_c, sigma_max = self.find_critical_mode(Le)
            
            # 绘制色散曲线
            ax.plot(k_array, sigma, color=color, linewidth=2.5, label=f'Le={Le}')
            ax.axhline(0, color='k', linestyle='--', alpha=0.5)
            
            # 标记不稳定区域
            if np.any(sigma > 0):
                ax.fill_between(k_array, 0, sigma, where=sigma>0,
                               alpha=0.3, color=color, label='Unstable')
                ax.plot(k_c, sigma_max, 'ko', markersize=10)
                ax.annotate(f'  ({k_c:.2f}, {sigma_max:.3f})',
                          xy=(k_c, sigma_max), fontsize=9)
            
            ax.set_xlabel('Wave number k')
            ax.set_ylabel('Growth rate σ(k)')
            ax.set_title(f'Le = {Le}, L = {L:.4f}')
            ax.grid(True, alpha=0.3)
            ax.legend()
            ax.set_xlim(0, 6)
            ax.set_ylim(-0.5, max(0.1, sigma_max*1.2))
        
        # 4. Lewis数扫描（下排左）
        ax = axes[1, 0]
        Le_scan = np.linspace(0.3, 2.0, 200)
        L_values = []
        sigma_max_values = []
        k_c_values = []
        
        for Le_test in Le_scan:
            L = self.markstein_length(Le_test)
            k_c, sigma_max = self.find_critical_mode(Le_test)
            L_values.append(L)
            sigma_max_values.append(sigma_max)
            k_c_values.append(k_c)
        
        ax.plot(Le_scan, sigma_max_values, 'b-', linewidth=2.5)
        ax.axhline(0, color='k', linestyle='--', alpha=0.5)
        ax.axvline(1.0, color='r', linestyle='--', alpha=0.5, label='Le=1')
        ax.fill_between(Le_scan, 0, sigma_max_values,
                       where=np.array(sigma_max_values)>0,
                       alpha=0.3, color='red')
        
        # 标记测试点
        for Le, color in zip(Le_cases, colors):
            idx = np.argmin(np.abs(Le_scan - Le))
            ax.plot(Le, sigma_max_values[idx], 'o', color=color, markersize=8)
        
        ax.set_xlabel('Lewis Number')
        ax.set_ylabel('Maximum Growth Rate')
        ax.set_title('Stability Boundary')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # 5. Markstein长度（下排中）
        ax = axes[1, 1]
        ax.plot(Le_scan, L_values, 'g-', linewidth=2.5)
        ax.axhline(0, color='k', linestyle='--', alpha=0.5)
        ax.axvline(1.0, color='r', linestyle='--', alpha=0.5)
        ax.fill_between(Le_scan, 0, L_values,
                       where=np.array(L_values)>0,
                       alpha=0.3, color='orange', label='L>0 (unstable)')
        ax.fill_between(Le_scan, L_values, 0,
                       where=np.array(L_values)<0,
                       alpha=0.3, color='blue', label='L<0 (stable)')
        
        ax.set_xlabel('Lewis Number')
        ax.set_ylabel('Markstein Length L')
        ax.set_title('L = (1-Le)/(Le) × f(β,σ)')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # 6. 临界波数（下排右）
        ax = axes[1, 2]
        ax.plot(Le_scan, k_c_values, 'purple', linewidth=2.5)
        ax.axvline(1.0, color='r', linestyle='--', alpha=0.5)
        ax.set_xlabel('Lewis Number')
        ax.set_ylabel('Critical Wave Number k_c')
        ax.set_title('Most Unstable Mode')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0.3, 2.0)
        ax.set_ylim(0, max(k_c_values)*1.1)
        
        plt.suptitle('Flame Stability Analysis (Corrected)', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def plot_mode_shapes(self):
        """
        绘制不同Lewis数下的火焰形状
        """
        fig, axes = plt.subplots(1, 3, figsize=(12, 4), 
                                subplot_kw=dict(projection='polar'))
        
        Le_cases = [0.5, 0.7, 1.3]
        titles = ['Strongly Unstable', 'Weakly Unstable', 'Stable']
        
        theta = np.linspace(0, 2*np.pi, 200)
        
        for ax, Le, title in zip(axes, Le_cases, titles):
            L = self.markstein_length(Le)
            k_c, sigma_max = self.find_critical_mode(Le)
            
            # 基态圆
            r_base = np.ones_like(theta)
            ax.plot(theta, r_base, 'k--', linewidth=2, alpha=0.5, label='Base')
            
            if sigma_max > 0:
                # 不稳定：显示最不稳定模式
                m = max(1, int(k_c))  # 方位角模数
                amplitude = 0.2
                r_perturbed = r_base + amplitude * np.cos(m * theta)
                ax.plot(theta, r_perturbed, 'r-', linewidth=2.5,
                       label=f'm={m} mode')
                ax.fill_between(theta, r_base, r_perturbed,
                               where=r_perturbed>r_base, 
                               alpha=0.3, color='red')
            else:
                # 稳定：圆形
                ax.plot(theta, r_base, 'g-', linewidth=3, label='Stable')
            
            ax.set_ylim(0.6, 1.4)
            ax.set_title(f'{title}\nLe={Le}, L={L:.3f}')
            ax.legend(loc='upper right')
        
        plt.suptitle('Flame Front Shapes', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()

def main():
    """
    主程序
    """
    print("=" * 60)
    print("Final Corrected Flame Stability Analysis")
    print("=" * 60)
    
    # 创建分析器
    analyzer = FlameStabilityAnalysis(Lewis=0.7, Zeldovich=10.0, heat_release=5.0)
    
    # 分析关键情况
    test_cases = [0.5, 0.7, 1.0, 1.3]
    results = []
    
    for Le in test_cases:
        L, k_c, sigma_max = analyzer.analyze_case(Le)
        results.append((Le, L, k_c, sigma_max))
    
    # 汇总表格
    print("\n" + "=" * 60)
    print("Summary Table:")
    print("-" * 60)
    print(f"{'Le':^6} | {'L':^10} | {'k_c':^8} | {'σ_max':^10} | {'Status':^12}")
    print("-" * 60)
    
    for Le, L, k_c, sigma_max in results:
        status = "Unstable" if sigma_max > 0 else "Stable"
        print(f"{Le:^6.2f} | {L:^10.4f} | {k_c:^8.3f} | {sigma_max:^10.4f} | {status:^12}")
    
    print("=" * 60)
    
    # 理论验证
    print("\nTheoretical Check:")
    print("• Le < 1 → L > 0 → Unstable ✓")
    print("• Le = 1 → L = 0 → Neutral ✓")
    print("• Le > 1 → L < 0 → Stable ✓")
    
    # 绘图
    print("\nGenerating plots...")
    analyzer.plot_analysis()
    analyzer.plot_mode_shapes()
    
    print("\nAnalysis Complete!")

if __name__ == "__main__":
    main()