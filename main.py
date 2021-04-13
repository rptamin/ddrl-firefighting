from madqn import MADQN
import pickle
import rlUtilities as rl
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # train and save a network
    algorithm = MADQN(mode='train')
    algorithm.train(num_episodes=1)

    # test a network or the heuristic
    # load_filename = 
    # test_method = 'network'
    # algorithm = MADQN(mode='test', filename=load_filename)
    # results = algorithm.test(num_episodes=1, method=test_method)
    test_method = 'network'
    algorithm = MADQN(mode='test', filename=None)
    results = algorithm.test(num_episodes=1, method=test_method)
    print(list(results))
    print(results)
    forest_states = results.get(0)
    forest_state = list(forest_states.get('sim_states'))[100]
    print(forest_state)
    image = rl.latticeforest_image(forest_state, (0,0), (100,100))
    plt.imshow(image)
    plt.show()
    # save the results to file
    # WARNING: the resulting output file may be very large, ~1.5 GB
    # save_filename = 'results_' + test_method + '.pkl'
    # output = open(save_filename, 'wb')
    # pickle.dump(results, output)
    # output.close()
