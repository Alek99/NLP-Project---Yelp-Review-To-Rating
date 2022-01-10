from simpletransformers.classification import ClassificationModel

class FinalModel():
    def __init__(self):
        self.model = ClassificationModel('xlnet', 'xlnet-base-cased', use_cuda=False, num_labels=5, args={'max_seq_length':256,'save_steps':1000, 'fp16':False, 'logging_steps': 1, 'train_batch_size':16, 'num_train_epochs':5})

    def run(self, df):
        self.model.train_model(df)

    def eval(self, df):
        return self.model.predict(df)

    def load(self, path, model_type):
        self.model = ClassificationModel(model_type, path, num_labels=5, use_cuda=False)




