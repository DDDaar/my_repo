import pandas as pd
import re
import os

# ==========================================
# 核心提取方法定义 (支持 A-G)
# ==========================================

def extract_prediction_method1(text):
    """方法1：多模式智能提取"""
    if pd.isna(text): return ""
    text_str = str(text).strip()
    if not text_str: return ""

    if "<answer>" in text_str and "</answer>" in text_str:
        start_idx = text_str.find("<answer>") + 8
        end_idx = text_str.find("</answer>", start_idx)
        if end_idx != -1:
            content = text_str[start_idx:end_idx].strip()
            if content: text_str = content

    single_letter_pattern = r'\b([A-G])\b\.?'
    match = re.search(single_letter_pattern, text_str, re.IGNORECASE)
    if match: return match.group(1).upper()

    patterns = [
        r'(?:answer|correct|right|choice)\s*(?:is|:|=)\s*([A-G])',
        r'([A-G])\s*(?:is|is the)?\s*(?:answer|correct|right|option)',
        r'option\s*([A-G])',
        r'choice\s*([A-G])',
        r'([A-G])\.\s',
        r'\b([A-G])\b',
    ]
    for pattern in patterns:
        match = re.search(pattern, text_str, re.IGNORECASE)
        if match:
            letter = match.group(1).upper()
            if letter in 'ABCDEFG':
                return letter

    for char in text_str:
        if char.isalpha() and char.upper() in 'ABCDEFG':
            return char.upper()
    return ""

def extract_prediction_method2(text):
    """方法2：首字母法"""
    if pd.isna(text): return ""
    text_str = str(text).strip()
    for char in text_str:
        if char.isalpha() and char.upper() in 'ABCDEFG':
            return char.upper()
    return ""

def extract_prediction_method3(text):
    """方法3：关键词定位法"""
    if pd.isna(text): return ""
    text_str = str(text).strip()
    keyword_pattern = r'(?:the\s+answer\s+is|final\s+answer\s+is|correct\s+option\s+is|答案是|最终答案是|正确选项是|所以选|结论是|最终选择为)\s*[:=：]?\s*([A-G])\b'
    
    match = re.search(keyword_pattern, text_str, re.IGNORECASE)
    if match: return match.group(1).upper()

    all_letters = re.findall(r'\b([A-G])\b', text_str, re.IGNORECASE)
    if all_letters: return all_letters[-1].upper()
    return ""

def extract_prediction(text, method=1):
    if method == 1: return extract_prediction_method1(text)
    if method == 2: return extract_prediction_method2(text)
    if method == 3: return extract_prediction_method3(text)
    return ""

# ==========================================
# 评估逻辑
# ==========================================

def run_evaluation(file_path):
    if not os.path.exists(file_path):
        print(f"错误：找不到文件 {file_path}")
        return

    try:
        df = pd.read_excel(file_path)
        
        print("\n" + "!"*80)
        print(f"{'数据集概况扫描':^76}")
        print("!"*80)
        
        # 预处理 hit 列
        if 'hit' in df.columns:
            df['hit_num'] = pd.to_numeric(df['hit'], errors='coerce').fillna(0)
            hit_mean = df['hit_num'].mean()
            print(f"总样本数: {len(df)}")
            print(f"原始 'hit' 列平均准确率: {hit_mean*100:.2f}%")
        else:
            print("错误：Excel 中缺少 'hit' 列，无法对比。")
            return

        # 真值清洗
        def clean_true_answer(x):
            if pd.isna(x): return ""
            m = re.search(r'([A-G])', str(x).upper())
            return m.group(1) if m else ""
        
        df['cleaned_answer'] = df['answer'].apply(clean_true_answer)
        methods = {1: "智能提取", 2: "首字母法", 3: "关键词法"}
        overall_results = {}

        for m_id, m_name in methods.items():
            print(f"\n{'='*80}")
            print(f">>> 评估方法 {m_id}: {m_name}")
            print(f"{'='*80}")

            # 计算预测
            df[f'pred_m{m_id}'] = df['prediction'].apply(lambda x: extract_prediction(x, m_id))
            
            # 判断是否正确 (排除真值为空的情况)
            df[f'correct_m{m_id}'] = (df[f'pred_m{m_id}'] == df['cleaned_answer']) & (df['cleaned_answer'] != "")
            
            # 1. 打印前5条基础信息 (预览)
            print(f"【前5条提取预览】")
            for i in range(min(5, len(df))):
                t, p = df['cleaned_answer'].iloc[i], df[f'pred_m{m_id}'].iloc[i]
                h = df['hit_num'].iloc[i]
                status = "✅" if t == p and t != "" else "❌"
                print(f"Idx: {i:<4} | True: {t:<2} | Pred: {p:<2} | 原Hit: {h} | 状态: {status}")

            # 2. 统计并展示恢复案例 (Hit=0 -> Correct)
            recovery_df = df[(df['hit_num'] == 0) & (df[f'correct_m{m_id}'] == True)]
            print(f"\n💡 恢复分析: 原始 Hit=0 但 [{m_name}] 提取正确的数量: {len(recovery_df)}")
            
            if not recovery_df.empty:
                print(f"--- 展示前 15 个恢复案例明细 ---")
                sample_recovery = recovery_df.head(15)
                for idx, row in sample_recovery.iterrows():
                    print(f"Index: {idx} | True Answer: {row['cleaned_answer']} | [{m_name}] Pred: {row[f'pred_m{m_id}']}")
                    # 打印原始模型输出的前150个字符，方便快速核对
                    raw_pred = str(row['prediction']).strip().replace('\n', ' ')
                    print(f"Raw Prediction: {raw_pred[:]}...") 
                    print("-" * 60)
            else:
                print("未发现恢复案例（该方法未能在 Hit=0 的样本中提取出正确答案）。")

            # 计算准确率
            valid_mask = df['cleaned_answer'] != ""
            acc = (df[f'correct_m{m_id}'].sum() / valid_mask.sum() * 100) if valid_mask.sum() > 0 else 0
            overall_results[m_name] = acc
            print(f"\n{m_name} 汇总准确率: {acc:.2f}%")

        # --- 最终汇总 ---
        print("\n\n" + "#" * 80)
        print(f"{'最终性能对比汇总':^75}")
        print("#" * 80)
        print(f"- {'原始文件 hit 列准确率':<30}: {hit_mean*100:.2f}%")
        for name, acc in overall_results.items():
            diff = acc - (hit_mean * 100)
            print(f"- {name:<35}: {acc:.2f}% (增益: {diff:+.2f}%)")
        print("#" * 80)

    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 请确保路径正确
    target_file = r'/home/ma-user/work/lbx/VLMEvalKit/outputs/Qwen2.5-VL-3B-Instruct/T20260203_Ga186fad8/Qwen2.5-VL-3B-Instruct_ScienceQA_TEST_exact_matching_result.xlsx'
    
    # target_file = r'/home/ma-user/work/lbx/VLMEvalKit/outputs/Qwen2.5-VL-7B-Instruct/T20260203_Ga186fad8/bak_20260203103608_ScienceQA_TEST/Qwen2.5-VL-7B-Instruct_ScienceQA_TEST_exact_matching_result.xlsx'
    run_evaluation(target_file)