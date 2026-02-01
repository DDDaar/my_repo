


import pandas as pd
import re
import os


# ==========================================
# 核心提取方法定义
# ==========================================

def extract_prediction_method1(text):
	"""
	方法1：从prediction列中提取答案字母
	处理多种格式：
	1. 如果存在<answer>标签，提取其中的内容
	2. 直接查找A-D字母
	"""
	if pd.isna(text):
		return ""

	text_str = str(text).strip()

	if not text_str:
		return ""

	# 先检查是否有<answer>标签
	if "<answer>" in text_str and "</answer>" in text_str:
		# 提取<answer>标签内容
		start_idx = text_str.find("<answer>")
		start_idx += 8  # len("<answer>")
		end_idx = text_str.find("</answer>", start_idx)
		if end_idx != -1:
			content = text_str[start_idx:end_idx].strip()
			if content:
				text_str = content

	# 在内容中搜索A-D字母
	# 方法1：尝试匹配单个大写字母（可能带句点）
	single_letter_pattern = r'\b([A-D])\b\.?'
	match = re.search(single_letter_pattern, text_str, re.IGNORECASE)
	if match:
		return match.group(1).upper()

	# 方法2：尝试匹配各种答案格式
	patterns = [
		r'(?:answer|correct|right|choice)\s*(?:is|:|=)\s*([A-D])',  # "answer is A"或"correct: B"
		r'([A-D])\s*(?:is|is the)?\s*(?:answer|correct|right|option)',  # "A is correct"或"B is the answer"
		r'option\s*([A-D])',  # "option A"
		r'choice\s*([A-D])',  # "choice B"
		r'([A-D])\.\s',  # "A. "开头的
		r'\b([A-D])\b',  # 最后的兜底，找任意位置的大写字母
	]

	for pattern in patterns:
		match = re.search(pattern, text_str, re.IGNORECASE)
		if match:
			letter = match.group(1).upper()
			# 验证字母在A-D范围内
			if letter in ['A', 'B', 'C', 'D']:
				return letter

	# 方法3：取第一个大写字母（作为最后的手段）
	for char in text_str:
		if char.isalpha() and char.upper() in ['A', 'B', 'C', 'D']:
			return char.upper()

	return ""


def extract_prediction_method2(text):
	"""
	方法2：直接提取预测的第一个字母
	简单地从文本中提取第一个大写字母A-D
	"""
	if pd.isna(text):
		return ""

	text_str = str(text).strip()

	if not text_str:
		return ""

	# 直接提取第一个大写字母A-D
	for char in text_str:
		if char.isalpha() and char.upper() in ['A', 'B', 'C', 'D']:
			return char.upper()

	return ""

def extract_prediction_method3(text):
	"""
	方法3：关键词定位法（中英文增强版）
	专门针对 "the answer is" 或 "答案是" 之后的字符
	"""
	if pd.isna(text): return ""
	text_str = str(text).strip()

	# 匹配中英文结论性短语
	# 包含：the answer is, final answer is, 答案是, 所以选, 结论是, 最终选择等
	keyword_pattern = r'(?:the\s+answer\s+is|final\s+answer\s+is|correct\s+option\s+is|答案是|最终答案是|正确选项是|所以选|结论是|最终选择为)\s*[:=：]?\s*([A-D])\b'

	match = re.search(keyword_pattern, text_str, re.IGNORECASE)
	if match:
		return match.group(1).upper()

	# 备选方案：如果没找到关键词，通常结论在文末，提取最后一个出现的 A-D
	all_letters = re.findall(r'\b([A-D])\b', text_str, re.IGNORECASE)
	if all_letters:
		return all_letters[-1].upper()

	return ""


def extract_prediction(text, method=1):
	"""提取函数主入口"""
	if method == 1: return extract_prediction_method1(text)
	if method == 2: return extract_prediction_method2(text)
	if method == 3: return extract_prediction_method3(text)
	return ""


# ==========================================
# 计算与统计逻辑
# ==========================================

def calculate_accuracy(file_path, extraction_method=1):
	method_name_map = {1: "多模式智能提取", 2: "直接提取首字母", 3: "中英文关键词定位"}
	method_name = method_name_map.get(extraction_method, "未知方法")

	try:
		df = pd.read_excel(file_path)

		# 提取预测答案
		df['extracted_prediction'] = df['prediction'].apply(lambda x: extract_prediction(x, extraction_method))

		# 提取真实答案（通常真实答案列比较干净，直接取第一个字母即可）
		def clean_true_answer(x):
			if pd.isna(x): return ""
			m = re.search(r'([A-D])', str(x).upper())
			return m.group(1) if m else ""

		df['true_answer_clean'] = df['answer'].apply(clean_true_answer)

		# 过滤出两者都成功提取的行进行比对
		valid_mask = (df['extracted_prediction'] != "") & (df['true_answer_clean'] != "")
		valid_df = df[valid_mask]

		correct_count = (valid_df['extracted_prediction'] == valid_df['true_answer_clean']).sum()
		total_valid = len(valid_df)
		accuracy = (correct_count / total_valid * 100) if total_valid > 0 else 0

		print(f"\n>> 运行结果 [{method_name}]:")
		print(f"   总样本数: {len(df)}")
		print(f"   有效提取数: {total_valid}")
		print(f"   正确数: {correct_count}")
		print(f"   准确率: {accuracy:.2f}%")

		# 保存结果
		output_file = file_path.replace('.xlsx', f'_results_method{extraction_method}.xlsx')
		#df.to_excel(output_file, index=False)
		return accuracy

	except Exception as e:
		print(f"处理出错: {e}")
		return None


# ==========================================
# 主程序
# ==========================================

if __name__ == "__main__":
	# 请根据实际路径修改文件名
	# file_path = r'C:\Users\李伯犀\Desktop\Qwen2.5-VL-3B-Instruct_ScienceQA_TEST_exact_matching_result——全部测试集sft.xlsx'
	file_path = r'C:\Users\李伯犀\Desktop\Qwen2.5-VL-7B-Instruct_ScienceQA_TEST_exact_matching_result.xlsx'

	if not os.path.exists(file_path):
		print("错误：找不到指定的Excel文件，请检查路径。")
	else:
		print("开始比对三种提取方法...")
		results = {}
		for m in [1, 2, 3]:
			acc = calculate_accuracy(file_path, extraction_method=m)
			results[m] = acc

		print("\n" + "=" * 40)
		print("最终准确率对比汇总：")
		print("-" * 40)
		names = {1: "智能提取", 2: "首字母法", 3: "中英关键词法"}
		for m, acc in results.items():
			if acc is not None:
				print(f"方法 {m} ({names[m]}): {acc:.2f}%")
		print("=" * 40)