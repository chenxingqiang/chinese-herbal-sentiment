import os
import json
import requests
import re
import argparse
from urllib.parse import quote
import time
from datetime import datetime
import random

# --- Configuration ---
# Replace with your SerpAPI key
SERPAPI_KEY = "ccb455bf2b78995c16bd150d248334ea8051214c1c76ce58f7582bc975638ee4"

# --- Core Functions ---

def search_google_scholar(query, num_results=5, start=0, max_retries=5):
    """
    Search Google Scholar using SerpAPI with enhanced error handling and retries
    
    Args:
        query (str): The search query
        num_results (int): Number of results to return
        start (int): Starting result index (for pagination)
        max_retries (int): Maximum number of retry attempts
        
    Returns:
        list: List of paper dictionaries or empty list if no results/error occurs
    """
    # Encode the query for URL
    encoded_query = quote(query)
    
    # SerpAPI endpoint
    url = f"https://serpapi.com/search.json?engine=google_scholar&q={encoded_query}&api_key={SERPAPI_KEY}&num={min(num_results, 20)}&start={start}"
    
    # Set up headers with a random user agent
    user_agents = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0.3 Safari/605.1.15',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36 Edg/91.0.864.59'
    ]
    
    # Initialize variables for retry logic
    retry_count = 0
    base_delay = 5  # Start with 5 seconds delay
    
    while retry_count < max_retries:
        # Add a small random delay between all requests
        time.sleep(random.uniform(2.0, 5.0))

        # Set up headers with a random user agent
        headers = {
            'User-Agent': random.choice(user_agents),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
        
        try:
            # Make the request with a generous timeout
            response = requests.get(url, headers=headers, timeout=60)
            
            # Check for rate limiting
            if response.status_code == 429:
                retry_after = int(response.headers.get('Retry-After', 60))
                print(f"  Rate limited. Server asks to retry after {retry_after} seconds.")
                time.sleep(retry_after)
                retry_count += 1
                continue
                
            response.raise_for_status()
            
            # Try to parse JSON
            try:
                data = response.json()
            except json.JSONDecodeError as e:
                print(f"  JSON decode error for query '{query}': {str(e)}")
                print(f"  Response content: {response.text[:500]}...")
                retry_count += 1
                # Add exponential backoff for JSON errors as well
                delay = base_delay * (2 ** retry_count)
                time.sleep(delay * random.uniform(0.8, 1.2))
                continue
            
            # Check for API errors
            if 'error' in data:
                print(f"  API error for query '{query}': {data.get('error', 'Unknown error')}")
                retry_count += 1
                delay = base_delay * (2 ** retry_count)
                time.sleep(delay * random.uniform(0.8, 1.2))
                continue
            
            # Check if there are organic results
            if 'organic_results' in data and data['organic_results']:
                return data['organic_results']
            else:
                print(f"  No results found for query: {query}")
                return []
                
        except requests.exceptions.RequestException as e:
            print(f"  Request error for query '{query}': {str(e)}")
            retry_count += 1
            if retry_count < max_retries:
                delay = base_delay * (2 ** retry_count)
                print(f"  Waiting {delay:.1f} seconds before retry...")
                time.sleep(delay * random.uniform(0.8, 1.2))
            continue
    
    # If we've exhausted all retries
    print(f"  Failed to fetch results for query: {query} after {max_retries} attempts")
    return []

def format_reference(paper):
    """
    Formats a reference entry for a single paper.

    Args:
        paper (dict): Paper information.

    Returns:
        str: Formatted reference entry.
    """
    title = paper.get('title', 'No Title')
    authors = ', '.join([author['name'] for author in paper.get('publication_info', {}).get('authors', [])])
    publication = paper.get('publication_info', {}).get('summary', 'No publication info')
    
    return f"{authors}. **{title}**. *{publication}*."

def analyze_relevance(paper, keywords):
    """
    Analyzes the relevance of a paper to a list of keywords.

    Args:
        paper (dict): Paper information.
        keywords (list): List of (keyword, weight) tuples.

    Returns:
        tuple: (relevance_score, matched_keywords)
    """
    score = 0
    matched_keywords = []
    
    title = paper.get('title', '').lower()
    snippet = paper.get('snippet', '').lower()
    
    # Check for keywords in title and snippet
    for keyword, weight in keywords:
        # Use regex for whole-word matching
        if re.search(r'\b' + re.escape(keyword.lower()) + r'\b', title):
            score += weight * 1.5  # Higher weight for keywords in the title
            matched_keywords.append(keyword)
        elif re.search(r'\b' + re.escape(keyword.lower()) + r'\b', snippet):
            score += weight
            matched_keywords.append(keyword)
            
    return score, list(set(matched_keywords))

def save_to_markdown(content, filename):
    """
    Saves content to a Markdown file.

    Args:
        content (str): The content to save.
        filename (str): The name of the file.
    """
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Saved to {filename}")

# --- Topic-specific Data Functions ---



def get_tcm_supply_chain_keywords():
    """
    Get English keywords related to TCM supply chain and consumer review analysis

    Returns:
    list: List of (keyword, weight) tuples
    """
    return [
        # Quality Evaluation
        ("quality evaluation", 10), ("quality control", 9), ("quality standard", 8), 
        ("authenticity identification", 7), ("heavy metal", 6), ("pesticide residue", 6), 
        ("active ingredient", 8), ("chemical fingerprint", 7), ("good agricultural practices", 5),
        # Processing
        ("processing technology", 9), ("decoction pieces", 7), ("extraction process", 8), 
        ("good manufacturing practice", 8), ("GMP", 8), ("production management", 7), 
        ("quality assurance", 8),
        # Logistics & Distribution
        ("logistics", 9), ("supply chain", 10), ("cold chain logistics", 7), 
        ("storage", 6), ("distribution efficiency", 7), ("traceability", 8), 
        ("supply chain visibility", 7), ("timeliness", 6),
        # Sales & Service
        ("e-commerce", 8), ("online to offline", 7), ("O2O", 7), ("price analysis", 6), 
        ("market analysis", 7), ("consumer behavior", 9), ("purchase intention", 8), 
        ("brand loyalty", 7),
        # After-sales Service
        ("after-sales service", 7), ("customer satisfaction", 8), ("complaint handling", 6), 
        ("customer feedback", 8), ("return policy", 5),
        # Sentiment Analysis
        ("sentiment analysis", 10), ("consumer reviews", 10), ("opinion mining", 9), 
        ("text mining", 8), ("natural language processing", 8), ("NLP", 8), 
        ("machine learning", 7), ("deep learning", 7), ("BERT", 8), ("LSTM", 7), 
        ("aspect-based sentiment analysis", 9)
    ]

def generate_tcm_search_queries():
    """
    Generate search queries for TCM supply chain and consumer review analysis

    Returns:
    list: List of search queries
    """
    return [
        '"traditional chinese medicine" AND "quality evaluation"', 
        '"herbal medicine" AND "quality control" AND "heavy metals"',
        '"TCM" AND "authenticity" AND "DNA barcoding"',
        '"chinese materia medica" AND "pesticide residue analysis"',
        '"quality standards for herbal medicines"',
        '"TCM processing technology" AND "optimization"',
        '"good manufacturing practice" AND "herbal medicine"',
        '"supply chain management" AND "traditional chinese medicine"',
        '"logistics traceability system" AND "TCM"',
        '"cold chain logistics for herbal products"',
        '"consumer behavior" AND "traditional chinese medicine"',
        '"e-commerce trends for herbal supplements"',
        '"market analysis of TCM industry"',
        '"brand strategy for traditional medicine"',
        '"customer satisfaction in TCM services"',
        '"sentiment analysis" AND "consumer reviews" AND "herbal medicine"',
        '"opinion mining" AND "TCM product reviews"',
        '"natural language processing" AND "patient feedback" AND "TCM"',
        '"aspect-based sentiment analysis of TCM reviews"',
        '"deep learning for text mining in healthcare reviews"',
        '"integrated quality control system for TCM supply chain"',
        '"from farm to pharmacy: TCM supply chain quality"',
        '"blockchain technology in TCM traceability"',
        '"big data analytics for TCM market trends"',
        '"consumer trust in online TCM pharmacies"',
        '"challenges in globalizing traditional chinese medicine"',
        '"regulatory standards for international trade of TCM"',
        '"sustainability in herbal medicine supply chain"',
        '"AI in traditional medicine quality assessment"',
        '"social media data mining for public perception of TCM"',
        '"impact of COVID-19 on TCM supply chain"',
        '"integration of TCM in modern healthcare systems"',
        '"pharmacovigilance for herbal medicines"',
        '"intellectual property rights for traditional medicine"'
    ]

# --- Markdown and Categorization Functions ---



def generate_detailed_markdown(analyzed_papers, keywords, search_queries, timestamp, topic_name):
    """
    Generates a detailed Markdown report of the search results.

    Args:
        analyzed_papers (list): The list of analyzed papers.
        keywords (list): The list of keywords used.
        search_queries (list): The list of search queries used.
        timestamp (str): The timestamp of the search.
        topic_name (str): The name of the research topic.

    Returns:
        str: The Markdown-formatted report.
    """
    markdown = f"# Google Scholar Search Results for {topic_name}\n\n"
    markdown += f"## Search Information\n\n"
    markdown += f"**Date:** {timestamp}\n\n"
    markdown += f"**Number of papers:** {len(analyzed_papers)}\n\n"
    markdown += f"**Search Queries:**\n\n"
    for query in search_queries:
        markdown += f"- {query}\n"
    markdown += f"\n**Keywords and Weights:**\n\n"
    for keyword, weight in keywords:
        markdown += f"- {keyword}: {weight}\n"
    markdown += f"\n## Top Relevant Papers\n\n"
    for i, (paper, score, matched_keywords) in enumerate(analyzed_papers, 1):
        reference = format_reference(paper)
        markdown += f"### {i}. {paper.get('title', 'No Title')}\n\n"
        markdown += f"**Reference:** {reference}\n\n"
        markdown += f"**Relevance Score:** {score}\n\n"
        markdown += f"**Matched Keywords:** {', '.join(matched_keywords)}\n\n"
        markdown += f"**Abstract:** {paper.get('snippet', 'No abstract available')}\n\n"
        if "link" in paper:
            markdown += f"**Link:** [{paper['link']}]({paper['link']})\n\n"
        if "cited_by" in paper and "value" in paper["cited_by"]:
            markdown += f"**Cited by:** {paper['cited_by']['value']} papers\n\n"
        markdown += f"---\n\n"
    return markdown

def get_tcm_topic_categories():
    """
    Gets the categories for the TCM topic.
    
    Returns:
        dict: A dictionary of category names and their associated keywords.
    """
    return {
        "Quality Evaluation": ["quality evaluation", "quality control", "authenticity identification", "heavy metal", "pesticide residue", "active ingredient", "chemical fingerprint"],
        "Processing": ["processing technology", "decoction pieces", "extraction process", "good manufacturing practice", "gmp", "production management"],
        "Logistics & Distribution": ["logistics", "supply chain", "cold chain logistics", "storage", "distribution efficiency", "traceability", "supply chain visibility"],
        "Sales & Service": ["e-commerce", "online to offline", "o2o", "price analysis", "market analysis", "consumer behavior", "purchase intention", "brand loyalty"],
        "After-sales Service": ["after-sales service", "customer satisfaction", "complaint handling", "customer feedback", "return policy"],
        "Sentiment Analysis": ["sentiment analysis", "consumer reviews", "opinion mining", "text mining", "natural language processing", "nlp", "machine learning", "deep learning", "bert", "lstm", "aspect-based sentiment analysis"]
    }

def generate_categorized_references(analyzed_papers, timestamp):
    """
    Generates a categorized list of references in Markdown format.

    Args:
        analyzed_papers (list): The list of analyzed papers.
        timestamp (str): The timestamp of the search.

    Returns:
        str: The Markdown-formatted categorized list.
    """
    categories = get_tcm_topic_categories()
    topic_name = "TCM Supply Chain and Consumer Review Analysis"
    
    categorized_papers = {category: [] for category in categories}
    
    for paper, score, keywords in analyzed_papers:
        title = paper.get('title', '').lower()
        snippet = paper.get('snippet', '').lower()
        text_to_check = title + ' ' + snippet
        
        for category, cat_keywords in categories.items():
            if any(re.search(r'\b' + re.escape(kw.lower()) + r'\b', text_to_check) for kw in cat_keywords):
                categorized_papers[category].append(paper)
                break  # 只分配到第一个匹配的类别

    markdown = f"# Categorized References for {topic_name}\n\n"
    markdown += f"**Generated on:** {timestamp}\n\n"
    
    for category, papers in categorized_papers.items():
        if papers:
            markdown += f"## {category}\n\n"
            for paper in papers:
                markdown += f"- {format_reference(paper)}\n"
            markdown += "\n"
            

def main():
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Search Google Scholar for papers on TCM Supply Chain and Consumer Review Analysis.")
    parser.add_argument("--num_results", type=int, default=100, help="Total number of papers to fetch.")
    parser.add_argument("--per_query", type=int, default=4, help="Number of papers to fetch per query.")
    parser.add_argument("--output_prefix", type=str, default="scholar_results", help="Prefix for the output files.")
    args = parser.parse_args()

    # --- Data Loading ---
    search_topic = "TCM Supply Chain and Consumer Review Analysis"
    print("Loading keywords and search queries for TCM...")
    keywords = get_tcm_supply_chain_keywords()
    search_queries = generate_tcm_search_queries()

    # --- Search Execution ---
    print(f"\n=== Starting Search: {search_topic} ===")
    print(f"Number of Search Queries: {len(search_queries)}")
    print(f"Target Paper Count: {args.num_results}")
    print(f"Papers per Query: {args.per_query}")

    all_papers = []
    processed_titles = set()

    for i, query in enumerate(search_queries, 1):
        if len(all_papers) >= args.num_results:
            print("\nTarget number of papers reached. Halting search.")
            break

        print(f"\n[{i}/{len(search_queries)}] Searching for: '{query}'")
        
        results = search_google_scholar(query, num_results=args.per_query)
        
        if results:
            new_papers_count = 0
            for paper in results:
                title = paper.get('title', '').strip().lower()
                if title and title not in processed_titles:
                    all_papers.append(paper)
                    processed_titles.add(title)
                    new_papers_count += 1
            print(f"  Found {len(results)} results, added {new_papers_count} new unique papers. Total: {len(all_papers)}")
        else:
            print(f"  No results found for this query.")

        # Add a random delay to avoid rate limiting
        time.sleep(random.uniform(3, 7))

    # --- Analysis and Sorting ---
    print("\nAnalyzing and sorting papers by relevance...")
    analyzed_papers = []
    for paper in all_papers:
        score, matched_keywords = analyze_relevance(paper, keywords)
        if score > 0:
            analyzed_papers.append((paper, score, matched_keywords))

    analyzed_papers.sort(key=lambda x: x[1], reverse=True)
    analyzed_papers = analyzed_papers[:args.num_results]

    # --- Summary Output ---
    print(f"\n--- Search Summary ---")
    print(f"Total unique papers found: {len(all_papers)}")
    print(f"Relevant papers found (score > 0): {len(analyzed_papers)}")
    print(f"Top 5 Relevant Papers:")
    for i, (paper, score, matched) in enumerate(analyzed_papers[:5], 1):
        print(f"  {i}. {paper.get('title', 'No Title')} (Score: {score:.2f}) - Matched: {', '.join(matched)}")

    # --- File Generation ---
    now = datetime.now()
    formatted_timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
    file_timestamp = now.strftime("%Y%m%d_%H%M%S")

    args.output_prefix = f"{args.output_prefix}_tcm"

    # Generate and save detailed results
    detailed_markdown = generate_detailed_markdown(analyzed_papers, keywords, search_queries, formatted_timestamp, search_topic)
    results_filename = f"{args.output_prefix}_{file_timestamp}.md"
    save_to_markdown(detailed_markdown, results_filename)

    # Generate and save reference list
    reference_markdown = "# References\n\n" + "\n".join([f"- {format_reference(p[0])}" for p in analyzed_papers])
    references_filename = f"{args.output_prefix}_references_{file_timestamp}.md"
    save_to_markdown(reference_markdown, references_filename)

    # Generate and save categorized references
    categorized_markdown = generate_categorized_references(analyzed_papers, formatted_timestamp)
    categorized_filename = f"{args.output_prefix}_categorized_{file_timestamp}.md"
    save_to_markdown(categorized_markdown, categorized_filename)

    # Save raw data to JSON
    data_filename = f"{args.output_prefix}_data_{file_timestamp}.json"
    with open(data_filename, "w", encoding="utf-8") as f:
        json.dump({ 
            "topic": "tcm",
            "topic_name": search_topic,
            "timestamp": formatted_timestamp,
            "search_queries": search_queries,
            "keywords": keywords,
            "papers": [p[0] for p in analyzed_papers]
        }, f, indent=2, ensure_ascii=False)

    # Save latest versions (without timestamp)
    save_to_markdown(detailed_markdown, f"{args.output_prefix}_search_results_latest.md")
    save_to_markdown(reference_markdown, f"{args.output_prefix}_references_latest.md")
    save_to_markdown(categorized_markdown, f"{args.output_prefix}_categorized_latest.md")
    with open(f"{args.output_prefix}_data_latest.json", "w", encoding="utf-8") as f:
        json.dump({
            "topic": "tcm",
            "topic_name": search_topic,
            "timestamp": formatted_timestamp,
            "search_queries": search_queries,
            "keywords": keywords,
            "papers": [p[0] for p in analyzed_papers]
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n=== Search Complete ===")
    print(f"- Detailed Results: {results_filename}")
    print(f"- References: {references_filename}")
    print(f"- Categorized References: {categorized_filename}")
    print(f"- Raw Data: {data_filename}")
    print(f"\nLatest versions also saved with the '_latest' suffix.")

if __name__ == "__main__":
    main()
