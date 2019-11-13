# import cv2
from rlif.rna import colorize_nucleotides
import numpy as np
import matplotlib.pyplot as plt
plt.ion()

################
### Test methods

def test_run(env):
    for i in range(100):
        env.reset()
        print(i+1)
        print(env.target_structure.seq)
        while not env.done:
            action = env.action_space.sample()
            _, _, env.done, _ = env.step(action)
            # print(env.solution.string, end='\r\r\r') #' ', env.target_structure.seq, '\n ',
            # time.sleep(0.05)
            # input()
        env.next_target_structure()
    env.reset()
    

def random_sampling_test(env,runs=1000):
    for _ in range(runs):
        env.reset()
        while not env.done:
            action = env.action_space.sample()
            _, _, env.done, _ = env.step(action)


def speed_test(env):
    for _ in range(1000):
        env.reset()
        while not env.done:
            action = env.action_space.sample()
            image, _, env.done, _ = env.step(action)


def visual_test(env, pause=False):
        name = 'Visual Test'
        # cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        # cv2.resizeWindow(name, 1000, 200)

        # driver = create_browser('double')

        for _ in range(20):
            env.reset()
            # show_rna(env.target_structure.seq, None, driver, 0, 'double')
            print(env.target_structure.struct_motifs)
            print(env.target_structure.seq)
            while not env.done:
                action = env.action_space.sample()
                image, _, env.done, _ = env.step(action)

                # image *= 120
                print(colorize_nucleotides(env.solution.string), end='\r')

                im = np.asarray(image, dtype=np.uint8).squeeze()
                # cv2.imshow(name, im); cv2.waitKey(1)
                plt.cla()
                plt.imshow(im); plt.show(); plt.pause(0.001)
                if pause: input()

            print(''.join([' '] * 500))
            env.solution.summary(True)
            print('\n')
            # show_rna(env.solution.folded_structure.seq, env.solution.string, driver, 1, 'double')
            if pause: input()
        cv2.destroyWindow(name)


def graph_visual_test(env, pause=True):
    name = 'Visual Test'
    env.dataset = Dataset(dataset='eterna', start=1, n_seqs=10)
    mappy = {'A':'green', 'U':'blue', 'G':'red', 'C':'purple', 'O':'black', '-':'gray'}
    step = 0
    for _ in range(20):
        env.reset()
        while not env.done:
            step += 1
            action = env.action_space.sample()
            state, _, env.done, _ = env.step(action)
            subgraph = env.solution.current_subgraph
            features = env.solution.get_features()
            machine = WeisfeilerLehmanMachine(subgraph, features, 2)
            processed = machine.extracted_features
            embedding = env.embedder.infer_vector(processed)
            print(colorize_nucleotides(env.solution.string), end='\r')

            # Show embeddings
            plt.figure(1); plt.cla()
            plt.ylim([-.8, .8])
            plt.plot(embedding); plt.show()

            # Show subgraph
            plt.figure(2)
            feats = [mappy[feat] for feat in features.values()]
            pos = nx.spring_layout(subgraph)
            nx.draw(subgraph, pos, node_color=feats)
            plt.show(); plt.pause(0.01)
            if pause: input()
            plt.cla()

        print(''.join([' '] * 500))
        env.solution.summary(True)
        print('\n')

if __name__ == "__main__":
    from rlif.learning import Trainer, get_parameters
    from rlif.environments import RnaDesign
    from rlif.rna import Dataset

    d = Dataset(start=1, n_seqs=100, dataset='eterna', encoding_type=2)
    e = RnaDesign(get_parameters('RnaDesign')['environment'])
    e.randomize = False
    e.dataset = d
    visual_test(e)
