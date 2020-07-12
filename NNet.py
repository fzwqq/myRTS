class NeuralNet():
    """
    This class specifies the base NeuralNet class. To define your own neural network,
    subclass this class and implement the functions below. The neural network does not
    consider the current player.
    """

    def __init__(self,):
        pass

    def train(self, examples):
        """
        This funciton trains the neural network with examples obtained from self-play.
        Input:
            example: a list of training examples, where each example is of form (state, , v),
            pi is the MCTS informed policy vector for the given board, and v is its value.
            The examples has board in its canonical form.
        :param examples:
        :return:
        """
        pass

    def predict(self, board):
        """
        :param board: current board in its canonical form.
        :return:
            pi: a policy vector for the current board - a numpy array of length actions size.
            v: a float in [-1, 1] that gives the value of the current board
        """
        pass

    def save_checkpoint(self, folder, filename):
        """
        Saves the current neural network (with its parameters) in folder/filename
        """

    def load_checkpoint(self, folder, filename):
        """
        Loads parameters of the neural network from folder/filename
        """
        pass