import json
import glob
import hashlib
# import logging
import pandas as pd
import networkx as nx
from tqdm import tqdm
from joblib import Parallel, delayed
# from parser import parameter_parser
import numpy.distutils.system_info as sysinfo
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import random
import matplotlib.pyplot as plt

# logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)

class WeisfeilerLehmanMachine:
    """
    Weisfeiler Lehman feature extractor class.
    """
    def __init__(self, graph, features, iterations):
        """
        Initialization method which also executes feature extraction.
        :param graph: The Nx graph object.
        :param features: Feature hash table.
        :param iterations: Number of WL iterations.
        """
        self.iterations = iterations
        self.graph = graph
        self.features = features
        self.nodes = self.graph.nodes()
        self.extracted_features = [str(v) for k,v in features.items()]
        self.do_recursions()

    def do_a_recursion(self):
        """
        The method does a single WL recursion.
        :return new_features: The hash table with extracted WL features.
        """
        new_features = {}
        for node in self.nodes:
            nebs = self.graph.neighbors(node)
            degs = [self.features[neb] for neb in nebs]
            features = "_".join([str(self.features[node])]+sorted([str(deg) for deg in degs]))
            hash_object = hashlib.md5(features.encode())
            hashing = hash_object.hexdigest()
            new_features[node] = hashing
        self.extracted_features = self.extracted_features + list(new_features.values())
        return new_features

    def do_recursions(self):
        """
        The method does a series of WL recursions.
        """
        for _ in range(self.iterations):
            self.features = self.do_a_recursion()
        
def dataset_reader(path):
    """
    Function to read the graph and features from a json file.
    :param path: The path to the graph json.
    :return graph: The graph object.
    :return features: Features hash table.
    :return name: Name of the graph.
    """
    name = path.strip(".json").split("/")[-1]
    data = json.load(open(path))
    graph = nx.from_edgelist(data["edges"])

    if "features" in data.keys():
        features = data["features"]
    else:
        features = nx.degree(graph)

    features = {int(k):v for k,v, in features.items()}
    return graph, features, name

def feature_extractor(path, rounds):
    """
    Function to extract WL features from a graph.
    :param path: The path to the graph json.
    :param rounds: Number of WL iterations.
    :return features: Document collection object.
    """
    graph, features, name = dataset_reader(path)
    machine = WeisfeilerLehmanMachine(graph,features,rounds)
    features = TaggedDocument(words = machine.extracted_features , tags = ["g_" + name])
    return features
        
def save_embedding(output_path, model, files, dimensions):
    """
    Function to save the embedding.
    :param output_path: Path to the embedding csv.
    :param model: The embedding model object.
    :param files: The list of files.
    :param dimensions: The embedding dimension parameter.
    """
    out = []
    for f in files:
        identifier = f.split("/")[-1].strip(".json")
        out.append([int(identifier)] + list(model.docvecs["g_"+identifier]))

    out = pd.DataFrame(out,columns = ["type"] +["x_" +str(dimension) for dimension in range(dimensions)])
    out = out.sort_values(["type"])
    out.to_csv(output_path, index = None)

def exctract_features(graph, features, rounds):
    
    machine = WeisfeilerLehmanMachine(graph, features, rounds)
    features = TaggedDocument(words = machine.extracted_features, tags=['None'])
    return features

def train(args):
    from rlif.environments import RnaGraphDesign
    from rlif.rna import Dataset
    from rlif.learning import get_parameters
    # Get graphs and features from the environment

    env = RnaGraphDesign(get_parameters('RnaGraphDesign'))
    d1 = Dataset(dataset='eterna', start=1, n_seqs=100)
    d2 = Dataset(dataset='rfam_learn_train', start=1, n_seqs=100)
    d3 = Dataset(dataset='rfam_learn_validation', start=1, n_seqs=100)
    d4 = Dataset(dataset='rfam_taneda', start=1, n_seqs=29)

    seqs = d1.sequences + d2.sequences + d3.sequences + d4.sequences

    env.dataset = Dataset(sequences=seqs)
    # env.dataset = Dataset(dataset='rfam_learn_train', start=1, n_seqs=500)
    # mappy = {'A':'green', 'U':'blue', 'G':'red', 'C':'purple', -1:'black', '-':'gray'}
    step = 0
    graphs = []
    for _ in range(329):
        env.reset()
        while not env.done:
            step += 1
            print(step)
            action = env.action_space.sample()
            _, _, env.done, _ = env.step(action)
            # print(colorize_nucleotides(self.solution.string), end='\r')
            subgraph = env.solution.current_subgraph
            features = env.solution.get_features()
            # graphs.append([subgraph, features])
            machine = WeisfeilerLehmanMachine(subgraph, features, args.wl_iterations)
            processed = machine.extracted_features
            feats = TaggedDocument(words = machine.extracted_features, tags=['None'])
            
            # feats = [mappy[feat] for feat in features.values()]
            # pos = nx.spring_layout(subgraph)
            # plt.cla()
            # # subgraph = nx.Graph()
            # # nx.set
            # nx.draw(subgraph, pos, node_color=feats)
            # plt.show(); plt.pause(0.01)
            # if pause: input()

    

    # subgraph_features = Parallel(n_jobs = args.workers)(delayed(exctract_features)(graph=g[0], features=g[1], rounds=args.wl_iterations) for g in tqdm(graphs))
    model = Doc2Vec(
        subgraph_features,
        size = args.dimensions,
        window = 0,
        min_count = args.min_count,
        dm = 0,
        sample = args.down_sampling,
        workers = args.workers,
        iter = args.epochs,
        alpha = args.learning_rate)
    # print(graphs[0])
    # print(graphs[0][1])
    # x, y, z, u = [], [], [], []
    # import numpy as np
    # res = np.zeros([200, 32])
    # for i in range(200):
    #     # feats = [value for value in graphs[i][1].values()]
    #     print(subgraph_features[i])
    #     res[i, :] = model.infer_vector(subgraph_features[i].words)
    #     # x.append(results[0])
    #     # y.append(results[1])
    #     # z.append(results[2])
    #     # u.append(results[3])

    # # print(x)
    # # print(y)
    # try:
    #     while True:
    #         f = int(input('First'))
    #         s = int(input('Second'))
    #         plt.cla()
    #         plt.scatter(res[:, f], res[:, s])
    #         # plt.scatter(x, y)
    #         input('next')
    # except KeyboardInterrupt:
    #     pass
    model.save('train500_4each512')
    # model.infer_vector(graphs[100][1])
    

def main(args):
    """
    Main function to read the graph list, extract features, learn the embedding and save it.
    :param args: Object with the arguments.
    """
    graphs = glob.glob(args.input_path + "*.json")
    print("\nFeature extraction started.\n")
    feature_collections = Parallel(n_jobs = args.workers)(delayed(exctract_features)(graph=g[0], features=g[1], rounds=args.wl_iterations) for g in tqdm(graphs))
    print("\nOptimization started.\n")
    
    model = Doc2Vec(feature_collections,
                    size = args.dimensions,
                    window = 0,
                    min_count = args.min_count,
                    dm = 0,
                    sample = args.down_sampling,
                    workers = args.workers,
                    iter = args.epochs,
                    alpha = args.learning_rate)
    
    model.save('model{}'.format(random.randint(1, 10000)))

    # save_embedding(args.output_path, model, graphs, args.dimensions)

def test(args):
    model = Doc2Vec.load('train500_4each512')
    from rlif.environments import RnaGraphDesign
    from rlif.rna import Dataset
    from rlif.learning import get_parameters
    # Get graphs and features from the environment

    env = RnaGraphDesign(get_parameters('RnaGraphDesign'))

    d1 = Dataset(dataset='eterna', start=1, n_seqs=100)
    d2 = Dataset(dataset='rfam_learn_train', start=1, n_seqs=100)
    d3 = Dataset(dataset='rfam_learn_validation', start=1, n_seqs=100)
    d4 = Dataset(dataset='rfam_taneda', start=1, n_seqs=29)

    seqs = d1.sequences + d2.sequences + d3.sequences + d4.sequences

    env.dataset = Dataset(sequences=seqs)

    # mappy = {'A':'green', 'U':'blue', 'G':'red', 'C':'purple', -1:'black', '-':'gray'}
    step = 0
    # graphs = []
    mappy = {'A':'green', 'U':'blue', 'G':'red', 'C':'purple', 'O':'black', '-':'gray'}
    for _ in range(10):
        env.reset()
        while not env.done:
            step += 1
            print(step)
            action = env.action_space.sample()
            _, _, env.done, _ = env.step(action)
            # print(colorize_nucleotides(self.solution.string), end='\r')
            subgraph = env.solution.current_subgraph
            features = env.solution.get_features()
            machine = WeisfeilerLehmanMachine(subgraph, features, args.wl_iterations)
            processed = machine.extracted_features
            embedding = model.infer_vector(processed)
            plt.figure(1)
            plt.cla()
            plt.ylim([-1, 1])
            plt.plot(embedding)
            plt.show()

            feats = [mappy[feat] for feat in features.values()]
            plt.figure(2)
            
            nx.draw(subgraph, node_color=feats)

            print(embedding)
            input()
            plt.cla()



if __name__ == "__main__":
    args = parameter_parser()
    # main(args)
    test(args)
    # test(args)
