import draw


class Stas:
    def __init__(self, name):
        self.name = name
        self.loss = []
        self.loss_x = []
        self.valid_pred = []
        self.valid_pred_x = []
        self.test_pred = []
        self.test_pred_x = []
        self.train_pred = []
        self.train_pred_x = []
        self.drawer = draw.drawer()

    def update(self, cnt, Loss=None, Valid_pred=None, Test_pred=None, Train_pred=None):
        if Loss != None:
            self.loss.append(Loss)
            self.loss_x.append(cnt)
        if Valid_pred != None:
            self.valid_pred.append(Valid_pred)
            self.valid_pred_x.append(cnt)
        if Test_pred != None:
            self.test_pred.append(Test_pred)
            self.test_pred_x.append(cnt)
        if Train_pred != None:
            self.train_pred.append(Test_pred)
            self.train_pred_x.append(cnt)

    def draw(self):
        self.drawer.draw(title=self.name, xlabel="batch",
                         x=self.loss_x, ylabel="loss", y=self.loss)
        self.drawer.draw(title=self.name, xlabel="batch",
                         x=self.valid_pred_x, ylabel="valid_pred", y=self.valid_pred)
        self.drawer.draw(title=self.name, xlabel="batch",
                         x=self.test_pred_x, ylabel="test_pred", y=self.test_pred)
        self.drawer.draw(title=self.name, xlabel="batch",
                         x=self.train_pred_x, ylabel="train_pred", y=self.train_pred)
