def data_gettype(filename):
    with open(filename) as file_object:
        lines = file_object.readlines()
    data_lines = []
    if 'assist' in filename:
        sep = '\t'
    else:
        sep = '\t'
    for i in range(len(lines)):
        if i % 3 == 1:
            data_lines.append(lines[i].rstrip().rstrip('\n').rstrip(',').split(sep))
    data_lines = sum(data_lines, [])
    return data_lines


def get_kc_set(path):
    if 'assist2009' in path:
        training_path = path + '/training.csv'
        testing_path = path + '/testing.csv'
    else:
        training_path = path + '/training.txt'
        testing_path = path + '/testing.txt'
    train_typeset = data_gettype(training_path)
    test_typeset = data_gettype(testing_path)
    type_set = train_typeset + test_typeset
    temp = {}
    temp = temp.fromkeys(type_set)
    type_set = list(temp.keys())
    type_length = len(type_set)
    type_trans = dict()
    for i in range(type_length):
        try:
            type_trans[int(type_set[i])] = i
        except:
            print()
    print(type_trans)
    return type_trans, type_length, train_typeset, test_typeset
