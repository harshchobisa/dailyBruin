import torch

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data):
        # save data as members of object
        self.data = data

        # do any processing of data

    def __getitem__(self, index):
        """
            Retrieves single data sample
            May do some processing

            Param :
                index : int 
            Return :
                tensor : data sample
        """
        # do any processing

        return data[index]

    def __len__(self):
        '''
        Return :
            int : length of dataset

        '''
        return len(self.data)


