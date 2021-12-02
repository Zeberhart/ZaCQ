from collections import defaultdict
import pickle

with open("../data/output/kw_results.pkl", "rb") as f:
    kw_results = pickle.load(f)
    
with open("../data/output/vo_results.pkl", "rb") as f:
    vo_results = pickle.load(f)
    
with open("../data/output/zacq_results.pkl", "rb") as f:
    zacq_results = pickle.load(f)
    
LANG_SIZE = {"java":33, "python":60,"javascript":4, "php":9, "ruby":7}
TOTAL_EVALS = sum(list(LANG_SIZE.values()))
LANG_PROPORTIONS = {language: LANG_SIZE[language]/TOTAL_EVALS for language in LANG_SIZE}


print("VO")
avgs = defaultdict(lambda: defaultdict(float))
results = vo_results

for language in results:
    for n in results[language][(0,0,0)]:
        for metric in results[language][(0,0,0)][n]:
            if "params" not in metric:
                avgs[n][metric] += results[language][(0,0,0)][n][metric]*LANG_PROPORTIONS[language]
                
for a in avgs:
    print(f"{a}: {str(avgs[a])}")
    

print("KW")
avgs = defaultdict(lambda: defaultdict(float))
results = kw_results

for language in results:
    for n in results[language]:
        for metric in results[language][n]:
            if "params" not in metric:
                avgs[n][metric] += results[language][n][metric]*LANG_PROPORTIONS[language]

for a in avgs:
    print(f"{a}: {str(avgs[a])}")

    
print("ZACQ")
avgs = defaultdict(lambda: defaultdict(float))
params_used = defaultdict(dict)
results = zacq_results

all_params = list(results["java"].keys())
for param_set in all_params:
    param_avgs = defaultdict(lambda: defaultdict(float))
    for language in results:
        for n in results[language][param_set]:
            for metric in results[language][param_set][n]:
                if "params" not in metric:
                    param_avgs[n][metric] += results[language][param_set][n][metric]*LANG_PROPORTIONS[language]
    for n in param_avgs:
        for metric in param_avgs[n]:
            if param_avgs[n][metric] > avgs[n][metric]:
                avgs[n][metric] = param_avgs[n][metric]
                params_used[n][metric] = param_set
                
for a in avgs:
    print(f"{a}: {str(avgs[a])}")

                