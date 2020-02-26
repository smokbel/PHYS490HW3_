import sys 
import numpy as np 
import json
import matplotlib.pyplot as plt 

def read_data(fname):
    with open(fname, 'r') as f:
        raw_data = f.read().split('\n')
    
    while len(raw_data[-1]) == 0:
        raw_data.pop()
    
    data = np.zeros((len(raw_data), len(raw_data[0])), dtype=np.int16)
    for (i, line) in enumerate(raw_data):
        for (j, char) in enumerate(line):
            data[i,j] = -1 if char == '-' else 1 
            
    return data 

def init_J(length):
    return np.random.choice([-1, 1], size=length)

def calc_H(s_vals, J):
    h = -np.sum(J * s_vals * np.roll(s_vals, -1, axis=1), axis=1)
    return h 


def calc_gradient(s_vals):
    return np.average(s_vals * np.roll(s_vals, -1, axis=1), axis=0)

def shuffle(data):
    ord = np.random.permutation(data.shape[0])
    return data[ord]

def calc_naive_avg(s_vals):
    transitions = s_vals * np.roll(s_vals, -1, axis=1)
    return np.average(transitions, axis=0)

def normalise(J):
    return J / (np.sum(np.absolute(J)) / (J.shape[0] // 2))


def main():
    np.random.seed(0)
    fname = sys.argv[1]
    learning_file = sys.argv[2]
    train_data = read_data(fname)
    
    with open(learning_file) as json_file:
        read_json = json.load(json_file)
    

    J = init_J(train_data.shape[1])
    
    B = read_json['B']
    lr = read_json['lr']
    num_iter = read_json['num_iter']
    errors = []
    b_pr_e = train_data.shape[0] // B 
    for e in range(num_iter):
        epoch_data = shuffle(train_data)
        for b in range(b_pr_e):
            batch = epoch_data[b*B:(b+1)*B]
            g = calc_gradient(batch)
            J = normalise(J + lr * g)
            error = np.average(calc_H(batch, J))
            errors.append(error)
            
    J_round = np.round(J,0)

    for i in range(len(J)):
        if J_round[i] == 0:
            J_round[i] = 1
        
    print('Final prediction:', '(0,1):', J_round[0],
                               '(1,2):', J_round[1], 
                               '(2,3):', J_round[2], 
                               '(3,4):', J_round[3] )
    plt.plot(errors)
    plt.xlabel('Training Batch')
    plt.ylabel('Hamiltonian Error')
    plt.savefig('plots/KLLoss.png')

if __name__ == '__main__':
    main()