import re
import json
from typing import TypedDict, Optional, Dict, Any
import openai
from iris_model import IrisModel

YANDEX_CLOUD_BASE_URL: str = "https://ai.api.cloud.yandex.net/v1"
YANDEX_CLOUD_FOLDER: Optional[str] = "b1gtnnrn79ee3ee7oai8" 
YANDEX_CLOUD_MODEL: Optional[str] = "aliceai-llm/latest"
YANDEX_CLOUD_API_KEY: Optional[str] = #

# Состояние — это схема (TypedDict/Pydantic/dataclass), представляющая общие данные графа.
#  Узлы читают состояние и возвращают его частичные обновления.
class AgentState(TypedDict, total=False):
	query: str
	use_tool: Optional[bool]
	final_answer: Optional[str]
	action: Optional[str]
	values: Optional[list]
	tool_result: Optional[str]

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
		return s

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

		return str(resp)
	
	except Exception as e:

		return f"LLM call failed: {e}"

def reasoning_node(state: AgentState) -> AgentState:

	req = state.get("query", "")

	prompt = (
		"У тебя есть инструмент `iris_predict`, который принимает ровно 4 числа "
		"[sepal_length, sepal_width, petal_length, petal_width] и возвращает метку вида.\n"
		"В ответе верни только JSON в одном из форматов:\n"
		"{\"action\":\"predict\", \"values\": [число, число, число, число]}\n"
		"или\n"
		"{\"action\":\"answer\", \"answer\": \"текст\"}.\n"
		"Если нужно вызвать модель — верни action=\"predict\" и values. Иначе верни answer.\n"
		"User query: " + req
	)
	resp = call_alisa_llm(prompt)

	try:
		m = re.search(r"\{.*\}", resp, re.S)
		if m:
			obj = json.loads(m.group(0))
			action = obj.get("action")
			if action == "predict":
				values = obj.get("values", [])
				if isinstance(values, list) and len(values) == 4 and all(isinstance(x, (int, float)) for x in values):
					return {**state, "use_tool": True, "values": values, "final_answer": None}
				else:
					return {**state, "use_tool": False, "final_answer": "LLM предложила некорректные значения для predict."}
			else:
				return {**state, "use_tool": False, "final_answer": obj.get("answer")}
	except Exception:
		pass

	return {**state, "use_tool": False, "final_answer": resp}

#Узел поиска: моделирует вызов модели и возвращает результат
def action_node(state: AgentState) -> AgentState:

	vals = state.get("values")
	if vals and isinstance(vals, list) and len(vals) == 4:
		model = IrisModel()
		pred = model.predict(vals)
		return {**state, "use_tool": False, "tool_result": pred, "final_answer": pred}

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
