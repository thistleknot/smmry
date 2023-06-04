#forked from https://github.com/TheCynosure/smmry_impl

import os
import re
import sys
from typing import Dict, List, Optional

class BSTNode:
    def __init__(self, data: str, word_len: int):
        self.data = data
        self.word_len = word_len
        self.score = 1
        self.left_child = None
        self.right_child = None

class Bst:
    def __init__(self):
        self.root = None

    def add(self, data: str, word_len: int) -> int:
        curr = self.root
        allocate = 1
        prev = None
        last_dir = None

        while curr is not None:
            shorter_word_len = min(curr.word_len, word_len)
            result = ord(data[0]) - ord(curr.data[0])
            if result == 0:
                curr.score += 1
                allocate = 0
                break
            elif result < 0:
                prev = curr
                curr = curr.left_child
                last_dir = 'left'
            else:
                prev = curr
                curr = curr.right_child
                last_dir = 'right'

        if allocate:
            node = BSTNode(data, word_len)
            if last_dir == 'left':
                prev.left_child = node
            elif last_dir == 'right':
                prev.right_child = node
            else:
                self.root = node
        return allocate

    def get_score(self, data: str, word_len: int) -> int:
        curr = self.root
        while curr is not None:
            shorter_word_len = min(curr.word_len, word_len)
            result = ord(data[0]) - ord(curr.data[0])
            if result == 0:
                return curr.score
            elif result < 0:
                curr = curr.left_child
            else:
                curr = curr.right_child

        return 0

    def get_node(self, data: str, word_len: int) -> Optional[BSTNode]:
        curr = self.root
        while curr is not None:
            shorter_word_len = min(curr.word_len, word_len)
            result = ord(data[0]) - ord(curr.data[0])
            if result == 0:
                return curr
            elif result < 0:
                curr = curr.left_child
            else:
                curr = curr.right_child

        return None

    def free_node(self, node: Optional[BSTNode]):
        if node is not None:
            self.free_node(node.left_child)
            self.free_node(node.right_child)
            node = None

    def free_bst(self) -> None:
        self.free_node(self.root)
        self.root = None

class SentenceNode:
    def __init__(self, data: str, score: int = 0):
        self.data = data
        self.link = None
        self.score = score

class LinkedList:
    def __init__(self):
        self.head = None
        self.tail = None
        self.size = 0

    def insert(self, data: str):
        node = SentenceNode(data)
        if self.head is None:
            self.head = node
        else:
            self.tail.link = node
        self.tail = node
        self.size += 1

    def free_list(self):
        current_node = self.head
        while current_node is not None:
            temp = current_node.link
            current_node.link = None
            current_node = temp
        self.head = None
        self.tail = None

def strip_newlines_tabs(text_buffer: str) -> str:
    return text_buffer.replace('\n', ' ').replace('\t', ' ')

def load_array(file_path: str) -> List[str]:
    with open(file_path, 'r') as f:
        lines = f.readlines()
        return [line.strip() for line in lines]

def is_title(text_buffer: str, word_len: int, titles: List[str]) -> bool:
    for title in titles:
        if len(title) == word_len - 1 and text_buffer.startswith(title):
            return True
    return False

def sentence_chop(text_buffer: str, titles: List[str]) -> LinkedList:
    sen_list = LinkedList()
    current_word_len = 0
    new_sentence = False
    last_sentence = 0
    inside_paren = False

    for i, c in enumerate(text_buffer):
        if c == ' ':
            current_word_len = 0
        elif c == '.':
            next_char = text_buffer[i+1] if i+1 < len(text_buffer) else None
            if not inside_paren and (
                next_char == ' ' or 
                next_char == '\0' or 
                next_char == '(' or
                next_char == '[') and not is_title(text_buffer[i - current_word_len:i], current_word_len, titles):

                new_sentence = True
                current_word_len = 0
                continue
        else:
            current_word_len += 1
            if c == '(':
                inside_paren = True
            elif c == ')':
                inside_paren = False

        if new_sentence:
            sen_list.insert(text_buffer[last_sentence:i].strip())
            last_sentence = i
            new_sentence = False

    if new_sentence:
        sen_list.insert(text_buffer[last_sentence:].strip())

    return sen_list

def load_list(file_path: str) -> Dict[str, str]:
    word_dict = {}

    with open(file_path, 'r') as f:
        for line in f.readlines():
            synonym, base_word = line.strip().split()
            base_word = base_word.lower()
            word_dict[synonym] = base_word

    return word_dict

def add_words_to_bst(word_bst: Bst, l: LinkedList, synonyms: Dict[str, str], irreg_nouns: Dict[str, str]):
    current_node = l.head
    while current_node is not None:
        c = current_node.data
        curr_word_start = 0
        curr_word_len = 0

        for i, char in enumerate(current_node.data):
            if char == ' ':
                word = current_node.data[curr_word_start:curr_word_start + curr_word_len]
                word = word[0].lower() + word[1:]
                
                if word in synonyms:
                    word = synonyms[word]
                elif word in irreg_nouns:
                    word = irreg_nouns[word]

                if word_bst.add(word, curr_word_len) == 0:
                    del(word)

                curr_word_start = i + 1
                curr_word_len = 0
            else:
                curr_word_len += 1

        current_node = current_node.link

def tally_top_scorers(word_bst: Bst, l: LinkedList, return_num: int, synonyms: Dict[str, str], irreg_nouns: Dict[str, str]) -> List[SentenceNode]:
    top_scorers = [None] * return_num
    current_node = l.head

    while current_node is not None:
        current_node.score = 0
        c = current_node.data
        curr_word_start = 0
        curr_word_len = 0

        for i, char in enumerate(current_node.data):
            if char == ' ':
                score = word_bst.get_score(current_node.data[curr_word_start:curr_word_start + curr_word_len], curr_word_len)
                if score > 1:
                    current_node.score += score

                curr_word_start = i + 1
                curr_word_len = 0
            else:
                curr_word_len += 1

        for i in range(return_num):
            if top_scorers[i] is None or top_scorers[i].score < current_node.score:
                top_scorers[i:] = [current_node] + top_scorers[i: -1]
                break

        current_node = current_node.link

    return top_scorers

def get_rid_of_simples(word_bst: Bst, simples: List[str]):
    for word in simples:
        node = word_bst.get_node(word, len(word))
        if node is not None:
            node.score = 0

def main(filename: str, num_sentences: int) -> None:
    with open(filename, 'r') as f:
        text_buffer = f.read()

    text_buffer = strip_newlines_tabs(text_buffer)
    titles = load_array("data/titles.txt")
    l = sentence_chop(text_buffer, titles)
    
    if num_sentences > l.size:
        print("Too many return sentences, not a long enough text!")
        sys.exit(1)

    synonyms = load_list("data/formattedcommonsyns.txt")
    irreg_nouns = load_list("data/formattedirregnouns.txt")
    word_bst = Bst()
    add_words_to_bst(word_bst, l, synonyms, irreg_nouns)

    simples = load_array("data/simplewords.txt")
    get_rid_of_simples(word_bst, simples)

    top_scorers = tally_top_scorers(word_bst, l, num_sentences, synonyms, irreg_nouns)

    for node in top_scorers:
        print(node.data)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: ./summ <filename> <# of sentences for summary>")
        sys.exit(1)

    filename = sys.argv[1]
    num_sentences = int(sys.argv[2])

    main(filename, num_sentences)