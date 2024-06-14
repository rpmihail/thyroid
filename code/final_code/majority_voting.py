split = 'val'

with open('results_unsampled_' + split + '.txt') as f:
    results = f.read().splitlines()

results = results[:-1]
predictions = {}
prob_predictions = {}

case = None
for result in results:
    cur_case = result.split(' ')[0]
    cur_case = cur_case.split('/')[-1]
    cur_case = cur_case.split('.')[0]
    cur_case = cur_case.split('_')[0]
    if case is None:
        comp = [0, 0, 0]
        echo = [0, 0, 0, 0]
        margin = [0, 0, 0]
        calc = [0, 0, 0, 0]
        
        comp_probs = [0.0, 0.0, 0.0]
        echo_probs = [0.0, 0.0, 0.0, 0.0]
        margin_probs = [0.0, 0.0, 0.0]
        calc_probs = [0.0, 0.0, 0.0, 0.0]
        case = cur_case
    if cur_case == case:
        pred_comp = int(result.split(' ')[1])
        comp[pred_comp] += 1
        pred_echo = int(result.split(' ')[2])
        echo[pred_echo] += 1
        pred_margin = int(result.split(' ')[3])
        margin[pred_margin] += 1
        pred_calc = int(result.split(' ')[4])
        calc[pred_calc] += 1

        comp_probs = [comp_probs[i] + float(result.split(' ')[5+i]) for i in range(3)]
        echo_probs = [echo_probs[i] + float(result.split(' ')[8+i]) for i in range(4)]
        margin_probs = [margin_probs[i] + float(result.split(' ')[12+i]) for i in range(3)]
        calc_probs = [calc_probs[i] + float(result.split(' ')[15+i]) for i in range(4)]
    else:
        predicted_classes = []
        predicted_classes.append(comp.index(max(comp)))
        predicted_classes.append(echo.index(max(echo)))
        predicted_classes.append(margin.index(max(margin)))
        predicted_classes.append(calc.index(max(calc)))
        predictions[case] = predicted_classes
        
        predicted_from_probs = []
        print(case, comp_probs, echo_probs, margin_probs, calc_probs)
        print(case, comp, echo, margin, calc)
        print(case, predicted_classes)
        predicted_from_probs.append(comp_probs.index(max(comp_probs)))
        predicted_from_probs.append(echo_probs.index(max(echo_probs)))
        predicted_from_probs.append(margin_probs.index(max(margin_probs)))
        predicted_from_probs.append(calc_probs.index(max(calc_probs)))
        prob_predictions[case] = predicted_from_probs
        print(case, comp_probs, echo_probs, margin_probs, calc_probs)
        print(case, predicted_from_probs)


        comp = [0, 0, 0]
        echo = [0, 0, 0, 0]
        margin = [0, 0, 0]
        calc = [0, 0, 0, 0]

        comp_probs = [0.0, 0.0, 0.0]
        echo_probs = [0.0, 0.0, 0.0, 0.0]
        margin_probs = [0.0, 0.0, 0.0]
        calc_probs = [0.0, 0.0, 0.0, 0.0]

        case = cur_case
        pred_comp = int(result.split(' ')[1])
        comp[pred_comp] += 1
        pred_echo = int(result.split(' ')[2])
        echo[pred_echo] += 1
        pred_margin = int(result.split(' ')[3])
        margin[pred_margin] += 1
        pred_calc = int(result.split(' ')[4])
        calc[pred_calc] += 1

        comp_probs = [comp_probs[i] + float(result.split(' ')[5+i]) for i in range(3)]
        echo_probs = [echo_probs[i] + float(result.split(' ')[8+i]) for i in range(4)]
        margin_probs = [margin_probs[i] + float(result.split(' ')[12+i]) for i in range(3)]
        calc_probs = [calc_probs[i] + float(result.split(' ')[15+i]) for i in range(4)]


predicted_classes = []
predicted_classes.append(comp.index(max(comp)))
predicted_classes.append(echo.index(max(echo)))
predicted_classes.append(margin.index(max(margin)))
predicted_classes.append(calc.index(max(calc)))
predictions[case] = predicted_classes
print(predictions)

predicted_from_probs = []
predicted_from_probs.append(comp_probs.index(max(comp_probs)))
predicted_from_probs.append(echo_probs.index(max(echo_probs)))
predicted_from_probs.append(margin_probs.index(max(margin_probs)))
predicted_from_probs.append(calc_probs.index(max(calc_probs)))
prob_predictions[case] = predicted_from_probs

print(prob_predictions)

ground_truth = {}
with open('../legacy_code/labels_stanford_' + split + '.txt') as f:
    gts = f.read().splitlines()

for gt in gts:
    data = gt.split(',')
    case = data[0].replace('_', '')
    case_gt = [int(data[16]), int(data[17]), int(data[18]), int(data[19])]
    ground_truth[case] = case_gt

print(ground_truth)

comp_correct = 0
echo_correct = 0
calc_correct = 0
margin_correct = 0
total = 0

for key in ground_truth.keys():
    gt = ground_truth[key]
    pred = predictions[key]
    if pred[0] == gt[0]:
        comp_correct += 1

    if pred[1] == gt[1]:
        echo_correct += 1

    if pred[2] == gt[2]:
        margin_correct += 1

    if pred[3] == gt[3]:
        calc_correct += 1

    total += 1

print(comp_correct/total, echo_correct / total, margin_correct / total, calc_correct/ total)



comp_correct = 0
echo_correct = 0
calc_correct = 0
margin_correct = 0
total = 0

for key in ground_truth.keys():
    gt = ground_truth[key]
    pred = prob_predictions[key]
    if pred[0] == gt[0]:
        comp_correct += 1

    if pred[1] == gt[1]:
        echo_correct += 1

    if pred[2] == gt[2]:
        margin_correct += 1

    if pred[3] == gt[3]:
        calc_correct += 1

    total += 1

print(comp_correct/total, echo_correct / total, margin_correct / total, calc_correct/ total)



