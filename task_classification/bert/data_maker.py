# -*- coding: utf-8 -*-
"""
@author: shiyi
@software: PyCharm
@file: data_maker.py
@time: 2022/6/25 16:35
"""

import json


accident_triggers = ['坠毁', '事故', '相撞', '爆炸', '火灾', '碰撞', '起火', '失踪', '出现意外', '决口']
deploy_triggers = ['部署', '驻扎', '进驻', '运输', '抵达']
exhibit_triggers = ['展示', '亮相', '参展', '公开', '曝光', '展出', '参观', '仪式', '航展',
                    '展览', '开放', '演示', '表演', '珠海航展', '公开亮相', '曝光｡', '显示',
                    '报道', '举行', '推出', '公布', '现身', '曝出', '发布', '出现', '展会',
                    '推销', '展销', '首次亮相', '宣传', '开放参观', '游行', '爆料', '公次开',
                    '对外开放', '露面', '亮相｡', '精彩表演', '博览会', '开幕', '首飞试飞',
                    '举办', '登场', ]
experiment_triggers = ['测试', '试射', '海试', '首飞', '试验', '试飞', '试航', '下水', '实验',
                       '首次飞行', '核试验', '发射', '验证', '试用']
indemnity_triggers = ['补给', '运送', '交付', '空中加油', '赠送']
manoeuvre_triggers = ['训练', '演习', '阅兵', '演练', '实兵演习', '实战演习', '特训',
                      '飞行训练', '联合演习', '军事演习', '军事训练', '军演', '比赛',
                      '军事训练', '军训', '比拼', '对抗', '对决', '考核', '联演', '比武考核']
support_triggers = ['护航', '提供支援', '支援', '营救', '搜救', '提供支援｡',
                    '救援', '救灾', '搜寻', '协助', '野营训练', '强化训练']
nonevent_triggers = ['15日', '动用', '车出', '展现', '巡航', '行军', '航行', '表明']


def load_json(path):
    with open(path, 'r', encoding='utf-8') as reader:
        js = json.load(reader)
    return js


def save_dict(dic, path):
    dic = {k: v for k, v in sorted(dic.items(), key=lambda item: item[1], reverse=True)}
    with open(path, 'w', encoding='utf-8') as writer:
        for k, v in dic.items():
            writer.write("{}\t{}\n".format(k, v))


def save_json(js):
    # with open(path, 'w', encoding='utf-8') as writer:
    #     for j in js:
    #         del j['tokens'], j['tokens_count'], j['entity_mention'], j['relation_mention']
    #         s = json.dumps(j, ensure_ascii=False)
    #         writer.write(s+"\n")
    accident = dict()
    deploy = dict()
    exhibit = dict()
    experiment = dict()
    indemnity = dict()
    manoeuvre = dict()
    support = dict()

    for j in js:
        del j['tokens'], j['tokens_count'], j['entity_mention'], j['relation_mention']
        if 'trigger' in j['event_mention']:
            text = j['event_mention']['trigger']['text']

            # 处理包含事件的数据
            if any(arg["role"] == "Content" for arg in j['event_mention']['arguments']):
                j['event_mention']['event_type'] = "Manoeuvre"
            elif any(arg["role"] == "Militaryforce" for arg in j['event_mention']['arguments']):
                j['event_mention']['event_type'] = "Deploy"
            elif any(arg["role"] == "Materials" for arg in j['event_mention']['arguments']):
                j['event_mention']['event_type'] = "Indemnity"
            elif any(arg["role"] == "Result" for arg in j['event_mention']['arguments']):
                j['event_mention']['event_type'] = "Accident"
            else:
                if text in accident_triggers:
                    j['event_mention']['event_type'] = "Accident"
                elif text in deploy_triggers:
                    j['event_mention']['event_type'] = "Deploy"
                elif text in exhibit_triggers:
                    j['event_mention']['event_type'] = "Exhibit"
                elif text in experiment_triggers:
                    j['event_mention']['event_type'] = "Experiment"
                elif text in indemnity_triggers:
                    j['event_mention']['event_type'] = "Indemnity"
                elif text in manoeuvre_triggers:
                    j['event_mention']['event_type'] = "Manoeuvre"
                elif text in support_triggers:
                    j['event_mention']['event_type'] = "Support"
                elif text in nonevent_triggers:
                    continue

            # 触发词统计
            if j['event_mention']['event_type'] == "Accident":
                accident[text] = accident.get(text, 0) + 1
            elif j['event_mention']['event_type'] == "Deploy":
                deploy[text] = deploy.get(text, 0) + 1
            elif j['event_mention']['event_type'] == "Exhibit":
                exhibit[text] = exhibit.get(text, 0) + 1
            elif j['event_mention']['event_type'] == "Experiment":
                experiment[text] = experiment.get(text, 0) + 1
            elif j['event_mention']['event_type'] == "Indemnity":
                indemnity[text] = indemnity.get(text, 0) + 1
            elif j['event_mention']['event_type'] == "Manoeuvre":
                manoeuvre[text] = manoeuvre.get(text, 0) + 1
            elif j['event_mention']['event_type'] == "Support":
                support[text] = support.get(text, 0) + 1

            with open("event_{}.json".format(j['event_mention']['event_type']), 'a', encoding='utf-8') as writer:
                s = json.dumps(j, ensure_ascii=False)
                writer.write(s + '\n')
        else:
            with open("Non-event.json", 'a', encoding='utf-8') as writer:
                s = json.dumps(j, ensure_ascii=False)
                writer.write(s + '\n')

    save_dict(accident, "trigger_Accident.txt")
    save_dict(deploy, "trigger_Deploy.txt")
    save_dict(exhibit, "trigger_Exhibit.txt")
    save_dict(experiment, "trigger_Experiment.txt")
    save_dict(indemnity, "trigger_Indemnity.txt")
    save_dict(manoeuvre, "trigger_Manoeuvre.txt")
    save_dict(support, "trigger_Support.txt")


dataset = load_json("../../datasets/Event_Competition/train_7000.json")
# save_json(dataset)
dataset += load_json("../../datasets/Event_Competition/valid_1500.json")
save_json(dataset)


# with open("helloworld.txt", 'w', encoding='utf-8') as writer:
#     writer.write("Hello, world!")
