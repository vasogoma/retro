rewards = [-0.06403333333333892, 27.088800000000763, 0.6524666666666692, 23.39023333333367, 0.30029999999998025, 3.17850000000022, -0.043299999999993684, 20.206966666666663, 2.0085000000000863, -0.4149666666666681, 3.210900000000167, 1.5691333333334, 20.582499999999992, 22.083966666666672, 23.29660000000023, 0.30029999999998025, 0.9341000000001611, 0.4538999999999952, 0.2703666666666661, 1.3992000000000122, 1.7016999999999969, 22.582533333333394, 1.4766000000001138, 20.822066666666665, 26.322866666667238, 0.704766666666834, -0.5055666666666742, 21.509733333333337, 0.9410333333333447, -0.4667999999998584, 26.779133333334023, 0.9669000000000062, 24.095133333333678, 20.29783333333333, 22.582533333333394, 0.808300000000007, 22.525000000000066, 3.17850000000022, 23.00620000000017, 21.30459999999999, -1.8292333333333308, 24.649833333333625, 24.398266666667027, 0.5674333333333196, 0.4933000000000061, 22.024200000000004, 5.752866666667428, 23.806033333333655, 0.9061000000000056, 21.53393333333333, 25.398766666667242, -0.6324333333333354, 0.30029999999998025, -0.9798666666666777, 22.083966666666672, 21.945800000000027, 1.5691333333334, 0.8436000000000003, 26.322866666667238, 23.806033333333655, 21.30459999999999, 21.94743333333333, 26.58326666666768, 1.518300000000009, 27.085666666667187, 21.94743333333333, 26.322866666667238, -0.4149666666666681, -0.4667999999998584, 26.322866666667238, 20.29783333333333, 1.448966666666669, 21.890266666666662, 0.22509999999999353, 21.153933333333327, 21.890266666666662, 24.649833333333625, 22.49873333333345, 1.5691333333334, 20.29783333333333, 1.5523000000000458, 23.806033333333655, -0.6611000000000083, 21.571466666666662, 1.9818000000000309, 0.4852666666666634, 29.83853333333281, 2.275333333333329, 23.77580000000029, 0.5178333333333378, 22.135200000000065, -0.3224333333333125, 20.974233333333327, -0.6611000000000083, -0.2712333333333396, 21.08510000000002, 20.29783333333333, 21.45046666666669, 23.509233333333526, 0.8648666666666722]
mean_reward = 12.162764000000116
victory_count = 0

for reward in rewards:
    if reward > 5:
        victory_count+=1

print(f"Victory count: {victory_count}")
