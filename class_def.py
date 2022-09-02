import pickle

class Foo(object):
    def __init__(self, name):
        self.name = name

def main():
    foo = Foo('a')
    with open('./tokenizer.pickle', 'wb') as f:
        pickle.dump([foo], f, -1)

if __name__=='__main__':
    main()