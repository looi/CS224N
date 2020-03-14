import csv, json
sub_dict = json.load(open('predictions_.json'))
with open('predictions.csv', 'w', newline='', encoding='utf-8') as csv_fh:
    csv_writer = csv.writer(csv_fh, delimiter=',')
    csv_writer.writerow(['Id', 'Predicted'])
    for uuid in sorted(sub_dict):
        csv_writer.writerow([uuid, sub_dict[uuid]])
