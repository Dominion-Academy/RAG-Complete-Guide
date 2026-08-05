from ragas import Dataset


dataset = Dataset(
    name="first_eval_dataset",
    backend="local/csv",
    root_dir=".",
)

records = [
    {
        "question": "В каком парке родилась идея талисмана Debug?",
        "reference": "В национальном парке Хванге (Hwange National Park), Зимбабве",
    },
    {
        "question": "Можно ли закрыть тикет #0000?",
        "reference": "Нет, ни при каких обстоятельствах и ни для одной роли, включая Администратора",
    },
]

for record in records:
    dataset.append(record)

dataset.save()
