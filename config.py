class DefaultConfig(object):
    print_freq = 1
    Epoch = 100

    train_batch_size = 128
    test_batch_size = 128

    term_numbers = 6
    cog_numbers = 2
    rule_numbers = term_numbers * cog_numbers
    weight_decay = 0.0005
    learning_rate = 0.03


    train_len = 5000

    max_len = 100


    lambda_o = 0
    lambda_w1 = 0
    lambda_w2 = 0
    dir = 'C:/Users/wsco70/Desktop/FDKT-TKDE/data/'
    # dir = '../data/'
    result_dir = '../result/'

    processing_num = 1
    load_model_path = None

    use_gpu = True

    small_data = False
    small_ratio = 20

    model_dir = '../code/checkpoints/'
    training_prediction_dir = '../code/training_result/'
    testing_prediction_dir = '../code/testing_result/'

    load_checkpoint = False
    code_input = False

    rnn_model = 'RNN'
