import json

def load_json(path):
    with open(path, 'r', encoding='utf-8') as reader:
        js = json.load(reader)
    return js


js_test_with_label = load_json("../../datasets/Event_Competition/submit_result_nagetive_0621_best.json")
js_test = load_json("../../datasets/Event_Competition/test_dataset_A.json")


n = len(js_test_with_label)
m = len(js_test)

assert m == n

for i in range(n):
    js_test[i]["event_mention"] = js_test_with_label[i]["event_mention"]


print(js_test)

f = json.dumps(js_test, ensure_ascii=False)
with open("test_A.json", 'w', encoding='utf-8') as writer:
    writer.write(f)


