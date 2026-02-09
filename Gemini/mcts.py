import math
import random

class Node:
    def __init__(self, prompt, score, depth, parent=None):
        self.prompt = prompt
        self.score = score
        self.depth = depth
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0.0

    def add_child(self, child_node):
        self.children.append(child_node)

    def update_value(self, reward):
        self.visits += 1
        self.value += reward

    def backpropagate(self, reward):
        self.update_value(reward)
        if self.parent:
            self.parent.backpropagate(reward)

    def uct(self, total_simulations, exploration_weight=math.sqrt(3)):
        if self.visits == 0:
            return float('inf')
        avg_value = self.value / self.visits
        exploration = exploration_weight * math.sqrt(math.log(total_simulations) / self.visits)
        return avg_value + exploration


class SharedNodeMCTS:
    def __init__(self):
        self.root = Node(prompt=None, score=0, depth=0)
        self.node_index = {(0, None): self.root}  # (depth, prompt): node
        self.total_simulations = 0

    def select_path_with_uct(self, depth):
        current_node = self.root
        path_scores = []
        for d in range(1, depth + 1):
            if not current_node.children:
                break
            current_node = self.select_node_by_uct(current_node)
            path_scores.append(current_node.score)
        return current_node, path_scores

    def expand_node(self, parent_node, prompt, score):
        if parent_node is None:
            parent_node = self.root
        depth = parent_node.depth + 1
        new_node = Node(prompt=prompt, score=score, depth=depth, parent=parent_node)
        parent_node.add_child(new_node)
        self.node_index[(depth, prompt)] = new_node
        return new_node

    def build_path(self, prompt_score_sequence):
        current = self.root
        for depth, (prompt, score) in enumerate(prompt_score_sequence, start=1):
            match = self.node_index.get((depth, prompt))
            if match:
                current = match
            else:
                current = self.expand_node(current, prompt, score)
        return current

    def backpropagate_reward(self, leaf_node, reward):
        self.total_simulations += 1
        leaf_node.backpropagate(reward)

    def select_node_by_uct(self, current_node):
        if not current_node.children:
            return current_node
        uct_values = [(child, child.uct(self.total_simulations)) for child in current_node.children]
        best_child = max(uct_values, key=lambda x: x[1])[0]
        return best_child

    def print_tree(self):
        print("=== MCTS Tree Structure ===")
        for (depth, prompt), node in sorted(self.node_index.items(), key=lambda x: x[0]):
            print(
                f"Depth {depth}, Prompt: {repr(prompt)[:30]}... | "
                f"Visits: {node.visits}, Value: {node.value:.2f}, "
                f"Children: {len(node.children)}"
            )