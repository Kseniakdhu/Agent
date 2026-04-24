import re
from typing import TypedDict, Optional, Dict, Any
import openai
from iris_model import IrisModel

YANDEX_CLOUD_BASE_URL: str = "https://ai.api.cloud.yandex.net/v1"
YANDEX_CLOUD_FOLDER: Optional[str] = "b1gtnnrn79ee3ee7oai8"
YANDEX_CLOUD_MODEL: Optional[str] = "aliceai-llm/latest"
#YANDEX_CLOUD_API_KEY: Optional[str] = 


# Состояние — это схема (TypedDict/Pydantic/dataclass), представляющая общие данные графа.
#  Узлы читают состояние и возвращают его частичные обновления.
class AgentState(TypedDict):
	query: str
	use_tool: Optional[bool]
	final_answer: Optional[str]


class Agent:
	def __init__(self):
		pass

	def invoke(self, state: Dict[str, Any]) -> Dict[str, Any]:
		s: AgentState = {
			"query": state.get("query", ""),
			"use_tool": False,
			"final_answer": None,
		}
		s = reasoning_node(s)
		if s.get("use_tool"):
			s = action_node(s)
		if s.get("final_answer") is None:
			s = reasoning_node(s)
		return s

# Узел внешнего инструмента
def call_alisa_llm(prompt: str) -> str:
	
	try:
		client = openai.OpenAI(
			api_key=YANDEX_CLOUD_API_KEY,
			base_url=YANDEX_CLOUD_BASE_URL,
			project=YANDEX_CLOUD_FOLDER,
		)
		resp = client.responses.create(
			model=f"gpt://{YANDEX_CLOUD_FOLDER}/{YANDEX_CLOUD_MODEL}",
			temperature=0.3,
			instructions="",
			input=prompt,
			max_output_tokens=500,
		)
		if hasattr(resp, "output_text") and resp.output_text:
			return resp.output_text
		if hasattr(resp, "output"):
			try:
				parts = []
				for item in resp.output:
					if isinstance(item, dict) and "content" in item:
						for c in item["content"]:
							if isinstance(c, dict) and c.get("type") == "output_text":
								parts.append(c.get("text", ""))
				if parts:
					return "\n".join(parts)
			except Exception:
				pass
		return str(resp)
	except Exception as e:
		return f"LLM call failed: {e}"

# Функция для извлечения 4 чисел из текста, если они есть
def parse_four_number(text: str):
	nums = re.findall(r"[0-9]*\.?[0-9]+", text)
	if len(nums) >= 4:
		try:
			vals = [float(n) for n in nums[:4]]
			return vals
		except Exception:
			return None
	return None

# Узел рассуждения: Понимает запрос и решает, или вызывать модель, или вызывает llm.
def reasoning_node(state: AgentState) -> AgentState:
	req = state.get("query", "")
	# Если распознаны 4 числа — попросить вызвать инструмент
	if parse_four_number(req):
		return {**state, "use_tool": True, "final_answer": None}
	# Иначе — спросить LLM и записать ответ
	answer = call_alisa_llm(req)
	return {**state, "use_tool": False, "final_answer": answer}

#Узел поиска: моделирует вызов модели и возвращает результат
def action_node(state: AgentState) -> AgentState:
	req = state.get("query", "")
	vals = parse_four_number(req)
	if vals:
		model = IrisModel()
		pred = model.predict(vals)
		return {**state, "use_tool": False, "final_answer": pred}
	# Если чисел нет, попробуем найти имя вида — если найдено, вернуть текст.
	match = re.search(r"(Iris[- ]\w+|setosa|versicolor|virginica|ирис [\w-]+)", req, re.I)
	if match:
		species = match.group(0)
		return {**state, "use_tool": False, "final_answer": f"Найден вид: {species}. Дополнительной информации в агенте нет."}
	# Ничего не найдено — инструмент отработал, но результата нет: вернём use_tool=False и пустой final_answer
	return {**state, "use_tool": False, "final_answer": None}

if __name__ == "__main__":
	print("Введите вариант запроса:\n" \
    "1) 4 числа: ширина и длина чашелистика, ширина и длина лепестка соответствено, для определения вида ириса\n" \
    "2) запрос о дополнительной информации о каком-то виде ириса.")
	a = Agent()
	while True:
		req = input("Запрос: ").strip()
		if not req:
			break
		res = a.invoke({"query": req})
		print("Результат:")
		print(res.get("final_answer"))
