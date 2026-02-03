# import pandas as pd
# import re
# import os

# # ==========================================
# # 核心提取方法定义 (支持 A-G)
# # ==========================================

# def extract_prediction_method1(text):
#     """方法1：多模式智能提取 (综合法)"""
#     if pd.isna(text): return ""
#     text_str = str(text).strip()
#     if not text_str: return ""

#     # 1. 优先提取 <answer> 标签内的内容
#     if "<answer>" in text_str and "</answer>" in text_str:
#         start_idx = text_str.find("<answer>") + 8
#         end_idx = text_str.find("</answer>", start_idx)
#         if end_idx != -1:
#             content = text_str[start_idx:end_idx].strip()
#             if content: text_str = content

#     # 2. 如果文本极短，直接匹配单个字母
#     single_letter_pattern = r'\b([A-G])\b\.?'
#     match = re.search(single_letter_pattern, text_str, re.IGNORECASE)
#     if match: return match.group(1).upper()

#     # 3. 常见模式匹配 (增加中文逻辑)
#     patterns = [
#         r'(?:answer|correct|right|choice)\s*(?:is|:|=)\s*([A-G])',
#         r'([A-G])\s*(?:is|is the)?\s*(?:answer|correct|right|option)',
#         r'option\s*([A-G])',
#         r'choice\s*([A-G])',
#         r'(?:故选|所以选|因此选|故选择|最终选择|正确选项是|答案是)\s*[:=：]?\s*([A-G])', # 同步增强方法1
#         r'([A-G])\.\s',
#         r'\b([A-G])\b',
#     ]
#     for pattern in patterns:
#         match = re.search(pattern, text_str, re.IGNORECASE)
#         if match:
#             letter = match.group(1).upper()
#             if letter in 'ABCDEFG':
#                 return letter

#     # 4. 最后的兜底：查找出现的字母
#     for char in text_str:
#         if char.isalpha() and char.upper() in 'ABCDEFG':
#             return char.upper()
#     return ""

# def extract_prediction_method2(text):
#     """方法2：首字母法"""
#     if pd.isna(text): return ""
#     text_str = str(text).strip()
#     for char in text_str:
#         if char.isalpha() and char.upper() in 'ABCDEFG':
#             return char.upper()
#     return ""

# def extract_prediction_method3(text):
#     """
#     方法3：关键词定位法 (已增强)
#     专门针对 'The answer is', '故选', '所以选' 等总结性语句进行提取
#     """
#     if pd.isna(text): return ""
#     text_str = str(text).strip()
    
#     # --- 修改核心：扩充了中文逻辑词 ---
#     # 包含：故选、故选择、因此选、所以选、答案是、最终答案是、正确选项是、结论是、最终选择为
#     # 兼容：冒号 (:, ：, =) 和 空格
#     keyword_pattern = r'(?:the\s+answer\s+is|final\s+answer\s+is|correct\s+option\s+is|答案是|最终答案是|正确选项是|所以选|故选|故选择|因此选|结论是|最终选择为)\s*[:=：]?\s*([A-G])\b'
    
#     match = re.search(keyword_pattern, text_str, re.IGNORECASE)
#     if match: return match.group(1).upper()

#     # 兜底：如果找不到关键词，返回文中出现的最后一个选项字母
#     # (很多CoT思维链是在最后给出答案的)
#     all_letters = re.findall(r'\b([A-G])\b', text_str, re.IGNORECASE)
#     if all_letters: return all_letters[-1].upper()
#     return ""

# def extract_prediction(text, method=1):
#     if method == 1: return extract_prediction_method1(text)
#     if method == 2: return extract_prediction_method2(text)
#     if method == 3: return extract_prediction_method3(text)
#     return ""

# # ==========================================
# # 评估逻辑
# # ==========================================

# def run_evaluation(file_path):
#     if not os.path.exists(file_path):
#         print(f"错误：找不到文件 {file_path}")
#         return

#     try:
#         df = pd.read_excel(file_path)
        
#         print("\n" + "!"*80)
#         print(f"{'数据集概况扫描':^76}")
#         print("!"*80)
        
#         # 预处理 hit 列
#         if 'hit' in df.columns:
#             df['hit_num'] = pd.to_numeric(df['hit'], errors='coerce').fillna(0)
#             hit_mean = df['hit_num'].mean()
#             print(f"总样本数: {len(df)}")
#             print(f"原始 'hit' 列平均准确率: {hit_mean*100:.2f}%")
#         else:
#             print("错误：Excel 中缺少 'hit' 列，无法对比。")
#             return

#         # 真值清洗
#         def clean_true_answer(x):
#             if pd.isna(x): return ""
#             m = re.search(r'([A-G])', str(x).upper())
#             return m.group(1) if m else ""
        
#         df['cleaned_answer'] = df['answer'].apply(clean_true_answer)
#         methods = {1: "智能提取", 2: "首字母法", 3: "关键词法(含故选/所以选)"}
#         overall_results = {}

#         for m_id, m_name in methods.items():
#             print(f"\n{'='*80}")
#             print(f">>> 评估方法 {m_id}: {m_name}")
#             print(f"{'='*80}")

#             # 计算预测
#             df[f'pred_m{m_id}'] = df['prediction'].apply(lambda x: extract_prediction(x, m_id))
            
#             # 判断是否正确 (排除真值为空的情况)
#             df[f'correct_m{m_id}'] = (df[f'pred_m{m_id}'] == df['cleaned_answer']) & (df['cleaned_answer'] != "")
            
#             # 1. 打印前5条基础信息 (预览)
#             print(f"【前5条提取预览】")
#             for i in range(min(5, len(df))):
#                 t, p = df['cleaned_answer'].iloc[i], df[f'pred_m{m_id}'].iloc[i]
#                 h = df['hit_num'].iloc[i]
#                 status = "✅" if t == p and t != "" else "❌"
#                 print(f"Idx: {i:<4} | True: {t:<2} | Pred: {p:<2} | 原Hit: {h} | 状态: {status}")

#             # 2. 统计并展示恢复案例 (Hit=0 -> Correct)
#             recovery_df = df[(df['hit_num'] == 0) & (df[f'correct_m{m_id}'] == True)]
#             print(f"\n💡 恢复分析: 原始 Hit=0 但 [{m_name}] 提取正确的数量: {len(recovery_df)}")
            
#             if not recovery_df.empty:
#                 print(f"--- 展示前 15 个恢复案例明细 ---")
#                 sample_recovery = recovery_df.head(15)
#                 for idx, row in sample_recovery.iterrows():
#                     print(f"Index: {idx} | True Answer: {row['cleaned_answer']} | [{m_name}] Pred: {row[f'pred_m{m_id}']}")
#                     # 打印原始模型输出的前150个字符，方便快速核对
#                     raw_pred = str(row['prediction']).strip().replace('\n', ' ')
#                     print(f"Raw Prediction: {raw_pred[:]}") 
#                     print("-" * 60)
#             else:
#                 print("未发现恢复案例（该方法未能在 Hit=0 的样本中提取出正确答案）。")

#             # 计算准确率
#             valid_mask = df['cleaned_answer'] != ""
#             acc = (df[f'correct_m{m_id}'].sum() / valid_mask.sum() * 100) if valid_mask.sum() > 0 else 0
#             overall_results[m_name] = acc
#             print(f"\n{m_name} 汇总准确率: {acc:.2f}%")

#         # --- 最终汇总 ---
#         print("\n\n" + "#" * 80)
#         print(f"{'最终性能对比汇总':^75}")
#         print("#" * 80)
#         print(f"- {'原始文件 hit 列准确率':<30}: {hit_mean*100:.2f}%")
#         for name, acc in overall_results.items():
#             diff = acc - (hit_mean * 100)
#             print(f"- {name:<35}: {acc:.2f}% (增益: {diff:+.2f}%)")
#         print("#" * 80)

#     except Exception as e:
#         import traceback
#         traceback.print_exc()

# if __name__ == "__main__":
#     # 请确保路径正确
#     # target_file = '/path/to/your/result.xlsx'
    
#     # 示例路径，请根据实际情况修改
#     target_file = '/home/ma-user/work/lbx/VLMEvalKit/outputs/Qwen2.5-VL-3B-Instruct/Qwen2.5-VL-3B-Instruct_RealWorldQA_exact_matching_result.xlsx'
    
#     run_evaluation(target_file)





import pandas as pd
import re
import os

# ==========================================
# 核心提取方法定义 (支持 A-G)
# ==========================================

def extract_prediction_method1(text):
    """方法1：多模式智能提取 (综合法)"""
    if pd.isna(text): return ""
    text_str = str(text).strip()
    if not text_str: return ""

    # 1. 优先提取 <answer> 标签内的内容
    if "<answer>" in text_str and "</answer>" in text_str:
        start_idx = text_str.find("<answer>") + 8
        end_idx = text_str.find("</answer>", start_idx)
        if end_idx != -1:
            content = text_str[start_idx:end_idx].strip()
            if content: text_str = content

    # 2. 如果文本极短，直接匹配单个字母
    single_letter_pattern = r'\b([A-G])\b\.?'
    match = re.search(single_letter_pattern, text_str, re.IGNORECASE)
    if match: return match.group(1).upper()

    # 3. 常见模式匹配
    patterns = [
        r'(?:answer|correct|right|choice)\s*(?:is|:|=)\s*([A-G])',
        r'([A-G])\s*(?:is|is the)?\s*(?:answer|correct|right|option)',
        r'option\s*([A-G])',
        r'choice\s*([A-G])',
        r'(?:故选|所以选|因此选|故选择|最终选择|正确选项是|答案是)\s*[:=：]?\s*([A-G])', 
        r'([A-G])\.\s',
        r'\b([A-G])\b',
    ]
    for pattern in patterns:
        match = re.search(pattern, text_str, re.IGNORECASE)
        if match:
            letter = match.group(1).upper()
            if letter in 'ABCDEFG':
                return letter

    # 4. 兜底：查找出现的字母
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
    """
    方法3：关键词定位法 (含中文逻辑)
    """
    if pd.isna(text): return ""
    text_str = str(text).strip()
    
    # 增强的关键词正则
    keyword_pattern = r'(?:the\s+answer\s+is|final\s+answer\s+is|correct\s+option\s+is|答案是|最终答案是|正确选项是|所以选|故选|故选择|因此选|结论是|最终选择为)\s*[:=：]?\s*([A-G])\b'
    
    match = re.search(keyword_pattern, text_str, re.IGNORECASE)
    if match: return match.group(1).upper()

    # 兜底：返回最后一个字母
    all_letters = re.findall(r'\b([A-G])\b', text_str, re.IGNORECASE)
    if all_letters: return all_letters[-1].upper()
    return ""

def extract_prediction(text, method=1):
    if method == 1: return extract_prediction_method1(text)
    if method == 2: return extract_prediction_method2(text)
    if method == 3: return extract_prediction_method3(text)
    return ""

# ==========================================
# 评估逻辑 (只针对 Hit=0 & Log含有Failed)
# ==========================================

def run_evaluation(file_path):
    if not os.path.exists(file_path):
        print(f"错误：找不到文件 {file_path}")
        return

    try:
        df = pd.read_excel(file_path)
        
        # 1. 基础数据准备
        if 'hit' not in df.columns:
            print("错误：缺少 'hit' 列")
            return
            
        # 确保 log 列存在，若不存在则填充空字符串以避免报错
        if 'log' not in df.columns:
            print("警告：缺少 'log' 列，将无法筛选 'Failed' 标记，默认仅筛选 hit=0")
            df['log'] = ""

        # 转换 hit 为数字
        df['hit_num'] = pd.to_numeric(df['hit'], errors='coerce').fillna(0)
        
        # 清洗真值 (Ground Truth)
        def clean_true_answer(x):
            if pd.isna(x): return ""
            m = re.search(r'([A-G])', str(x).upper())
            return m.group(1) if m else ""
        
        df['cleaned_answer'] = df['answer'].apply(clean_true_answer)
        
        # 2. 确定筛选范围
        # 条件：Hit 为 0  且  Log 包含 "Failed" (忽略大小写)
        mask_failed_target = (df['hit_num'] == 0) & (df['log'].astype(str).str.contains("Failed", case=False, na=False))
        
        # 统计基础信息
        total_samples = len(df)
        original_hits = df[df['hit_num'] == 1].shape[0] # 原本就对的数量
        target_failed_count = mask_failed_target.sum() # 符合处理条件的错误数量
        
        print("\n" + "!"*80)
        print(f"{'数据集概况':^76}")
        print("!"*80)
        print(f"总样本数: {total_samples}")
        print(f"原始正确数 (Hit=1): {original_hits} ({(original_hits/total_samples*100):.2f}%)")
        print(f"待处理错误样本 (Hit=0 & Log含有'Failed'): {target_failed_count}")

        if target_failed_count == 0:
            print("没有符合条件的错误样本，程序结束。")
            return

        methods = {1: "智能提取", 2: "首字母法", 3: "关键词法(含故选)"}
        final_stats = {}

        # 3. 循环评估方法
        for m_id, m_name in methods.items():
            print(f"\n{'='*80}")
            print(f">>> 方法 {m_id}: {m_name}")
            print(f"{'='*80}")
            
            # 仅提取所有预测值 (为了方便计算，虽然我们只关心 failed 的，但全量跑也很快)
            df[f'pred_m{m_id}'] = df['prediction'].apply(lambda x: extract_prediction(x, m_id))
            
            # 计算是否匹配真值
            df[f'is_match_m{m_id}'] = (df[f'pred_m{m_id}'] == df['cleaned_answer']) & (df['cleaned_answer'] != "")
            
            # *** 核心筛选：只看 原本Failed 且 现在Match 的行 ***
            # recovered_mask = (在目标错误范围内) AND (新方法提取正确)
            recovered_mask = mask_failed_target & df[f'is_match_m{m_id}']
            recovered_df = df[recovered_mask]
            
            recovered_count = len(recovered_df)
            
            print(f"🔍 成功挽回数量: {recovered_count} / {target_failed_count} (在该部分错误中的挽回率: {(recovered_count/target_failed_count*100 if target_failed_count else 0):.2f}%)")
            
            if recovered_count > 0:
                print(f"\n--- 挽回案例明细 (仅展示前 20 条) ---")
                # 打印详细信息
                for idx, row in recovered_df.head(20).iterrows():
                    t_ans = row['cleaned_answer']
                    p_val = row[f'pred_m{m_id}']
                    raw_pred = str(row['prediction']).strip().replace('\n', ' ')
                    
                    print(f"[Row {idx}] 真值: {t_ans} | 提取: {p_val}")
                    # 截取原始预测文本的前100个字符显示，避免刷屏
                    print(f"   原始预测(前150字符): {raw_pred[:150]}...")
                    print("-" * 60)
            else:
                print("   (无挽回案例)")

            # 计算最终修正后的总准确率
            # 逻辑：原始正确数 + 本次挽回数 (因为假设原始Hit=1的依然正确)
            final_acc = (original_hits + recovered_count) / total_samples * 100
            final_stats[m_name] = {
                "recovered": recovered_count,
                "final_acc": final_acc
            }

        # 4. 最终汇总表
        print("\n\n" + "#" * 80)
        print(f"{'最终效果对比':^75}")
        print("#" * 80)
        print(f"{'基准 (原始 Hit=1)':<30} | 准确率: {(original_hits/total_samples*100):.2f}%")
        print("-" * 80)
        for name, data in final_stats.items():
            print(f"{name:<30} | 挽回: {data['recovered']:<4} | 修正后准确率: {data['final_acc']:.2f}%")
        print("#" * 80)

    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 请修改为您的实际文件路径
    target_file = '/home/ma-user/work/lbx/VLMEvalKit/outputs/Qwen2.5-VL-3B-Instruct/Qwen2.5-VL-3B-Instruct_VStarBench_exact_matching_result.xlsx'
    
    run_evaluation(target_file)