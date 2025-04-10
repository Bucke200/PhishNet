import re
import numpy as np
import pandas as pd
from urllib.parse import urlparse
import tldextract
from collections import Counter
import Levenshtein
import math
# Note: TFIDFVectorizer and StandardScaler are typically used in the preprocessing/training script, not here.

def shannon_entropy(text):
    """Calculate Shannon entropy of a string"""
    if not text:
        return 0
    entropy = 0
    # Calculate probability for each character
    char_counts = Counter(text)
    text_len = float(len(text))
    for count in char_counts.values():
        p_x = count / text_len
        if p_x > 0:
            entropy += - p_x * math.log(p_x, 2)
    return entropy

def comprehensive_phishing_features(url):
    """Extracts a comprehensive set of features from a given URL."""
    features = {}

    # Handle potential non-string input gracefully
    if not isinstance(url, str):
        url = str(url) # Attempt to convert to string
        if not isinstance(url, str): # If conversion failed
             print(f"Warning: Could not process non-string input: {url}")
             # Return empty dict, preprocessing script should handle this row later
             return {}

    original_url = url # Keep original for some checks if needed

    # --- Default values in case parsing fails ---
    scheme = 'http' # Default scheme
    domain = ''
    path = ''
    query = ''
    fragment = ''
    extracted_subdomain = ''
    extracted_domain = ''
    extracted_suffix = ''
    # -------------------------------------------

    # Add scheme if missing (often needed for parsing)
    if not re.match(r'^https?://', url, re.IGNORECASE):
        if '.' in url and '/' not in url.split('.', 1)[0]:
             url = 'http://' + url
        # If it still doesn't look like a URL that urlparse/tldextract can handle,
        # the try/except block below will catch it.

    try:
        # Attempt parsing
        # Use cache_dir=None or specify a writable path if default causes issues
        # Consider tldextract.TLDExtract(cache_dir=False) if caching is problematic
        extracted = tldextract.extract(url)
        parsed = urlparse(url)

        # Assign parsed components if successful
        scheme = parsed.scheme if parsed.scheme else 'http'
        # Prefer urlparse netloc if available, otherwise reconstruct from tldextract
        domain = parsed.netloc if parsed.netloc else f"{extracted.subdomain}.{extracted.domain}.{extracted.suffix}".strip('.')
        path = parsed.path if parsed.path else ''
        query = parsed.query if parsed.query else ''
        fragment = parsed.fragment if parsed.fragment else ''
        extracted_subdomain = extracted.subdomain
        extracted_domain = extracted.domain
        extracted_suffix = extracted.suffix

        # Ensure domain is consistent if urlparse failed to get it but tldextract worked
        if not domain and extracted.fqdn:
             domain = extracted.fqdn

        # Ensure domain doesn't include scheme after parsing
        if domain.startswith(scheme + '://'):
             domain = domain[len(scheme + '://'):]

    except Exception as e: # Catch broader exceptions during parsing
        print(f"Warning: Could not parse URL '{original_url}' effectively: {e}. Using defaults/basic extraction.")
        # Use basic string splitting as fallback if parsing failed
        # scheme remains default 'http'
        parts = original_url.split('/', 1)
        domain = parts[0] # Basic assumption
        path = '/' + parts[1] if len(parts) > 1 else ''
        query = '' # Reset query/fragment as they are unreliable
        fragment = ''
        # Reset tldextract parts to empty strings as they couldn't be extracted
        extracted_subdomain = ''
        extracted_domain = ''
        extracted_suffix = ''

    #---- FEATURE EXTRACTION (using variables defined in try or except block) ----#

    # Basic length features
    features['url_length'] = len(original_url)
    features['domain_length'] = len(domain)
    features['path_length'] = len(path)

    # Character distribution (use original_url for full context)
    features['count_dots'] = original_url.count('.')
    features['count_hyphens'] = original_url.count('-')
    features['count_underscores'] = original_url.count('_')
    features['count_slashes'] = original_url.count('/')
    features['count_questionmarks'] = original_url.count('?')
    features['count_equals'] = original_url.count('=')
    features['count_ats'] = original_url.count('@')
    features['count_ampersands'] = original_url.count('&')
    features['count_exclamations'] = original_url.count('!')
    features['count_spaces'] = original_url.count(' ')
    features['count_www'] = original_url.lower().count('www')
    features['count_com'] = original_url.lower().count('.com')

    # Digit and letter ratios (use original_url)
    digits = sum(c.isdigit() for c in original_url)
    letters = sum(c.isalpha() for c in original_url)
    features['digit_count'] = digits
    features['letter_count'] = letters
    features['digit_letter_ratio'] = digits / (letters + 1)

    # Domain specific features
    ip_match = re.match(r'^(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})(:[0-9]+)?$', domain)
    features['has_ip_address'] = 1 if ip_match else 0
    features['has_port'] = 1 if ':' in domain else 0
    features['subdomain_count'] = extracted_subdomain.count('.') + 1 if extracted_subdomain else 0
    features['is_https'] = 1 if scheme == 'https' else 0 # Added protocol feature

    # Path features
    features['path_depth'] = path.count('/')
    features['has_suspicious_path'] = 1 if path and re.search(r'(login|admin|verify|secure|account)', path.lower()) else 0

    # Query string features
    features['query_length'] = len(query)
    features['query_param_count'] = len(query.split('&')) if query else 0

    # Security terms features (use original_url)
    security_terms = ['secure', 'account', 'webscr', 'login', 'ebayisapi', 'sign', 'banking', 'confirm']
    features['security_terms_count'] = sum(1 for term in security_terms if term in original_url.lower())

    # Shannon Entropy (use original_url)
    features['entropy'] = shannon_entropy(original_url)

    # Enhanced length metrics
    features['fragment_length'] = len(fragment)
    domain_tokens = [token for token in re.split(r'[.-]', domain) if token]
    features['domain_token_count'] = len(domain_tokens)
    path_tokens = [token for token in path.split('/') if token]
    features['path_token_count'] = len(path_tokens)

    # Advanced character distribution (use original_url)
    url_len_or_1 = len(original_url) if len(original_url) > 0 else 1
    features['special_char_ratio'] = sum(not c.isalnum() for c in original_url) / url_len_or_1
    features['upper_case_ratio'] = sum(c.isupper() for c in original_url) / url_len_or_1
    features['vowel_ratio'] = sum(c.lower() in 'aeiou' for c in original_url) / url_len_or_1
    features['consonant_ratio'] = sum(c.lower() in 'bcdfghjklmnpqrstvwxyz' for c in original_url) / url_len_or_1

    # Statistical features (use original_url)
    char_counts = Counter(original_url)
    features['unique_char_ratio'] = len(char_counts) / url_len_or_1

    # Shannon entropy variants
    domain_entropy = shannon_entropy(domain)
    path_entropy = shannon_entropy(path)
    query_entropy = shannon_entropy(query)
    features['domain_entropy'] = domain_entropy
    features['path_entropy'] = path_entropy
    features['query_entropy'] = query_entropy
    url_entropy = features['entropy']
    features['entropy_ratio'] = domain_entropy / (url_entropy if url_entropy > 0 else 1)

    # URL shortener detection
    shorteners = ['bit.ly', 'tinyurl.com', 't.co', 'goo.gl', 'is.gd', 'cli.gs', 'ow.ly',
                 'buff.ly', 'adf.ly', 'tiny.cc', 'lnkd.in', 'db.tt', 'qr.ae', 'cutt.ly']
    registered_domain_full = f"{extracted_domain}.{extracted_suffix}".strip('.')
    features['is_shortened'] = 1 if registered_domain_full in shorteners else 0

    # Domain-specific features using tldextract results
    features['domain_has_digit'] = 1 if any(c.isdigit() for c in extracted_domain) else 0
    features['subdomain_has_digit'] = 1 if any(c.isdigit() for c in extracted_subdomain) else 0
    features['tld'] = extracted_suffix # Keep TLD as string for now, handle in preprocessing
    features['has_suspicious_tld'] = 1 if extracted_suffix in ['tk', 'ml', 'ga', 'cf', 'gq', 'xyz'] else 0

    # Security indicator character combinations (use original_url)
    sec_patterns = ['secure', 'account', 'update', 'login', 'verify', 'password', 'confirm']
    for pattern in sec_patterns:
        features[f'contains_{pattern}'] = 1 if pattern in original_url.lower() else 0

    # Brand protection - compare with popular targets
    popular_brands = ['paypal', 'apple', 'amazon', 'microsoft', 'google', 'facebook', 'netflix', 'bank']
    min_distances = []
    domain_lower = domain.lower()
    for brand in popular_brands:
        # Direct presence
        features[f'contains_{brand}'] = 1 if brand in domain_lower else 0

        # Levenshtein distance against domain parts
        domain_parts = [part for part in re.split(r'[.-]', domain_lower) if part] # Ensure non-empty parts
        if domain_parts:
            try:
                 min_dist = min(Levenshtein.distance(brand, part) for part in domain_parts)
                 min_distances.append(min_dist)
                 features[f'{brand}_min_distance'] = min_dist
            except ValueError:
                 features[f'{brand}_min_distance'] = -1
        else:
             features[f'{brand}_min_distance'] = -1

    if min_distances:
        features['min_brand_distance'] = min(min_distances)
    else:
         features['min_brand_distance'] = -1

    # Suspicious patterns (use original_url)
    features['has_multiple_subdomains'] = 1 if extracted_subdomain.count('.') >= 1 else 0
    features['has_suspicious_chars'] = 1 if re.search(r'xn--|%[0-9A-Fa-f]{2}', original_url) else 0
    features['has_at_symbol'] = 1 if '@' in original_url else 0
    features['has_hexadecimal'] = 1 if re.search(r'0x[0-9a-fA-F]+', original_url) else 0
    features['has_ip_pattern'] = 1 if re.search(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', original_url) else 0

    # Path and query analysis
    features['dir_count'] = path.count('/')
    path_dirs = [d for d in path.split('/') if d] # Filter out empty strings
    features['avg_dir_length'] = np.mean([len(d) for d in path_dirs]) if path_dirs else 0
    # Check for common executable/script extensions in path
    suspicious_extensions = ['.php', '.exe', '.js', '.cgi', '.pl', '.sh', '.py', '.asp', '.aspx', '.dll']
    features['has_file_extension_in_path'] = 1 if any(ext in path.lower() for ext in suspicious_extensions) else 0

    # Character sequence analysis (use original_url)
    url_len = len(original_url)
    for n in [2, 3]:  # bigrams and trigrams
        if url_len >= n:
            ngram_counts = Counter(original_url[i:i+n] for i in range(url_len - n + 1))
            # Avoid division by zero for very short URLs
            denominator = (url_len - n + 1)
            features[f'unique_{n}gram_ratio'] = len(ngram_counts) / denominator if denominator > 0 else 0
        else:
            features[f'unique_{n}gram_ratio'] = 0 # Handle short URLs

    return features

# Example usage (optional, can be removed or kept for testing)
if __name__ == '__main__':
    sample_urls = [
        'br-icloud.com.br',
        'mp3raid.com/music/krizz_kaliko.html',
        'bopsecrets.org/rexroth/cr/1.htm',
        'http://buzzfil.net/m/show-art/ils-etaient-loin-de-s-imaginer-que-le-hibou-allait',
        'espn.go.com/nba/player/_/id/3457/brandon-rush',
        'yourbittorrent.com/?q=anthony-hamilton-soulife',
        'allmusic.com/album/crazy-from-the-heat-r16990',
        'corporationwiki.com/Ohio/Columbus/frank-s-benson-P3333917.aspx',
        'myspace.com/video/vid/30602581',
        'https://repl-mess.myfreesites.net/', # Added from previous discussion
        'http://192.168.1.1/admin', # Example IP address URL
        'http://example.com/%7Euser', # Example URL encoding
        'badurl<script>alert(1)</script>.com' # Example with potentially problematic chars
    ]

    all_features = []
    for url in sample_urls:
        print(f"\nProcessing URL: {url}")
        extracted_features = comprehensive_phishing_features(url)
        # Only attempt to add if features were extracted
        if extracted_features:
            print(extracted_features)
            all_features.append(extracted_features)
        else:
            print("Skipping this URL due to processing error.")

    # Optional: Convert to DataFrame for better viewing
    # Requires pandas to be installed
    if all_features: # Only create DataFrame if list is not empty
        try:
            import pandas as pd
            # Ensure consistent columns even if some features were missing for certain URLs
            df_features = pd.DataFrame(all_features).fillna(0) # Fill missing features with 0
            print("\n--- Features DataFrame ---")
            print(df_features)
        except ImportError:
            print("\nPandas not installed. Cannot display as DataFrame.")
        except Exception as e:
            print(f"\nError creating DataFrame: {e}")
    else:
        print("\nNo features were successfully extracted.")
