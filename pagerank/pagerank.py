import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    # Initialize probability distribution dictionary and set number of pages and links in page
    prob_dist = {page_name: 0 for page_name in corpus}
    num_pages = len(corpus)
    num_links = len(corpus[page])

    # If page has no links return equal probability for corpus
    if len(corpus[page]) == 0:
        for page_name in prob_dist:
            prob_dist[page_name] = 1 / num_pages
        return prob_dist

    # Probability of picking page at random and picking a link at random from a page
    random_prob = (1 - damping_factor) / num_pages
    link_prob = damping_factor / num_links

    # Adding each probability to the distribution
    for page_name in prob_dist:
        prob_dist[page_name] += random_prob
        if page_name in corpus[page]:
            prob_dist[page_name] += link_prob

    return prob_dist


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    # Storing page ranks and choosing a page at random
    page_rank = {}
    next_page = random.choice(list(corpus))

    # Iterate over the rest of the samples
    for i in range(n-1):
        # Create a transition model each time with the probability of each page and choose the next page at random considering transition model
        model = transition_model(corpus, next_page, damping_factor)
        next_page = random.choices(list(model), weights=model.values(), k=1).pop()

        # Fill page rank with the number of times page is visited
        if next_page in page_rank:
            page_rank[next_page] += 1
        else:
            page_rank[next_page] = 1

    # Normalize the number of visits
    for page in page_rank:
        page_rank[page] = page_rank[page] / n

    return page_rank


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    num_pages = len(corpus)
    random_prob = (1 - damping_factor) / num_pages
    init_prob = 1 / num_pages
    iteration = 0

    # Initial page rank
    page_rank = {page_name: init_prob for page_name in corpus}
    new_rank = {page_name: None for page_name in corpus}
    max_change = init_prob

    # Iterate until change < 0.001
    while max_change > 0.001:
        iteration += 1
        max_change = 0

        for page_name in corpus:
            choice_prob = 0
            for other_page in corpus:
                # If page has no links pick randomly from corpus
                if len(corpus[other_page]) == 0:
                    choice_prob += page_rank[other_page] * init_prob
                # If page has links pick randomly from all links
                elif page_name in corpus[other_page]:
                    choice_prob += page_rank[other_page] / len(corpus[other_page])
            # Calculating new ranks of page
            rank = random_prob + (damping_factor * choice_prob)
            new_rank[page_name] = rank

        # Normaliza new page ranks, like in sample_pagerank
        norm_factor = sum(new_rank.values())
        new_rank = {page: (rank / norm_factor) for page, rank in new_rank.items()}

        # Find the max change in page ranks
        for page_name in corpus:
            change = abs(page_rank[page_name] - new_rank[page_name])
            if change > max_change:
                max_change = change

        # Update the page ranks with the new ranks
        page_rank = new_rank.copy()

    return page_rank


if __name__ == "__main__":
    main()
