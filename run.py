from agent import agent_graph, State, load_model
from langchain.messages import HumanMessage


def main():
    if not load_model():
        print("Модель не найдена или не загрузилась"); raise SystemExit(1)
    state = State(message=[])
    print("Запуск")
    while True:
        try:
            user_input = input("Введите сообщение: ")
        except (KeyboardInterrupt, EOFError):
            print("\nВыход.")
            break

        state['message'].append(HumanMessage(content=user_input))

        try:
            result = agent_graph.invoke(state)
        except Exception as e:
            print("Ошибка при выполнении агента:", e)
            continue

        out = result.get('message', result) if isinstance(result, dict) else result
        if hasattr(out, 'content'):
            print(out.content)
        elif isinstance(out, list) and out and hasattr(out[0], 'content'):
            print('\n'.join(getattr(m, 'content', str(m)) for m in out))
        else:
            print(out)


if __name__ == '__main__':
    main()
