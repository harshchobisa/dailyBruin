import argparse
import asyncio
from datetime import datetime
import json
import os.path
import re
from urllib.request import Request, urlopen

from bs4 import BeautifulSoup
import pandas as pd


ARTICLE_CATEGORIES = [
    "a-closer-look",
    "academics",
    "academics-news",
    "ae",
    "ae-columns",
    "ae-infocus",
    "ae-spectrum",
    "arts-entertainment",
    "athletics-news",
    "baseball",
    "beach-volleyball",
    "behind-the-bruin",
    "box-office",
    "bruin-bucks",
    "bruingastronomer",
    "california",
    "campus",
    "campus-quad",
    "cartoons",
    "club-im-sports",
    "community",
    "copy-shop",
    "crime",
    "cross-country",
    "editorials",
    "enterprise",
    "features",
    "film-tv",
    "fire",
    "football",
    "graphics",
    "gymnastics",
    "higher-education",
    "illo",
    "infocus",
    "interactive-page",
    "international",
    "lifestyle",
    "lifestyle-quad",
    "los-angeles",
    "los-angeles-city-council",
    "mens-basketball",
    "mens-golf",
    "mens-soccer",
    "mens-tennis",
    "mens-volleyball",
    "mens-water-polo",
    "music",
    "national",
    "news",
    "news-infocus",
    "news-spectrum",
    "opinion",
    "opinion-columns",
    "podcasts",
    "press-pass",
    "prime",
    "quad",
    "rowing",
    "science-health",
    "softball",
    "spectrum",
    "sports",
    "sports-columns",
    "sports-spectrum",
    "student-government",
    "student-life",
    "swim-dive",
    "theater-arts",
    "throwback-thursday",
    "track-and-field",
    "transportation",
    "uncategorized",
    "university-of-california",
    "video",
    "westwood",
    "westwood-quad",
    "womens-basketball",
    "womens-golf",
    "womens-soccer",
    "womens-tennis",
    "womens-volleyball",
    "womens-water-polo",
]


def parse_shell_args():
    """Parses shell arguments."""

    parser = argparse.ArgumentParser(
        description="Process Daily Bruin data into a single CSV file."
    )
    parser.add_argument(
        "--csv_one",
        type=str,
        required=True,
        metavar="file",
        help="CSV file containing page view data",
    )
    parser.add_argument(
        "--json_one",
        type=str,
        required=True,
        metavar="file",
        help="JSON file containing article data",
    )
    parser.add_argument(
        "--csv_two",
        type=str,
        required=True,
        metavar="file",
        help="CSV file containing page view data",
    )
    parser.add_argument(
        "--json_two",
        type=str,
        required=True,
        metavar="file",
        help="JSON file containing article data",
    )
    parser.add_argument(
        "--output", type=str, required=True, metavar="file", help="Name of output file",
    )
    parser.add_argument(
        "--force", action="store_true", help="Whether to overwrite existing CSV file.",
    )

    return parser.parse_args()


def clean_articles_dict(articles_dict):
    """Cleans articles dictionary and returns a DataFrame"""

    articles_df = pd.DataFrame.from_dict(articles_dict["rss"]["channel"]["item"])[
        ["title", "link", "post_date", "creator", "encoded", "category"]
    ]

    # 1. Creator
    articles_df.creator = articles_df.creator.map(lambda x: x["__cdata"])

    # 2. Encoded text
    def clean_html(html):
        html = re.sub(r"<.*?>", " ", str(html))
        html = re.sub(r"\n", " ", html)
        html = re.sub(r"\\", " ", html)
        return html

    articles_df.encoded = articles_df.encoded.map(lambda x: x[0]["__cdata"])
    articles_df.encoded = articles_df.encoded.map(clean_html)
    articles_df = articles_df.rename(columns={"encoded": "encoded_text"})

    # 3. Authors and categories
    def parse_authors_and_categories(category_items):
        authors = []
        categories = []
        for d in category_items:
            if d["_domain"] == "author":
                authors.append(d["_nicename"])
            if d["_domain"] == "category":
                categories.append(d["_nicename"])
        return {"authors": authors, "categories": categories}

    authors_and_categories = articles_df.category.map(parse_authors_and_categories)
    articles_df["authors"] = [ac["authors"] for ac in authors_and_categories]
    articles_df["categories"] = [ac["categories"] for ac in authors_and_categories]
    articles_df = articles_df.drop("category", axis=1)

    # 4. Link
    def parse_url_path(url):
        regex = r"https://(?:.*.)?dailybruin.com(\/.*)"
        matches = re.match(regex, url)
        if matches is None:
            raise ValueError(f"clean_articles_dict: invalid url: {url}")
        return matches.group(1)

    articles_df["page"] = articles_df.link.map(parse_url_path)

    # 5. Date
    articles_df["date"] = articles_df.post_date.map(
        lambda s: datetime.fromisoformat(str(s["__cdata"]))
    )
    articles_df = articles_df.drop("post_date", axis=1)

    return articles_df


def clean_page_views_df(page_views_df):
    """Cleans page views DataFrame."""

    page_views_df = page_views_df[["Page", "Pageviews"]].rename(
        columns={"Page": "page", "Pageviews": "views"}
    )
    page_views_df = page_views_df.dropna()

    # 1. Page
    def clean_page_path(path):
        if "?" in path:
            path = path.split("?")[0]
        if path[0] != "/":
            path = "/" + path
        if path[-1] != "/":
            path += "/"
        return path

    page_views_df.page = page_views_df.page.map(clean_page_path)

    # 2. Views
    page_views_df.views = page_views_df.views.map(
        lambda s: int(str(s).replace(",", ""))
    )

    return page_views_df


async def create_merged_df(page_views_df, articles_df):
    """Creates final merged DataFrame."""

    merged_df = pd.merge(
        page_views_df, articles_df, how="outer", left_on="page", right_on="page"
    )
    merged_df = merged_df.drop_duplicates(subset=["page"]).dropna()
    merged_df = merged_df.reset_index(drop=True)

    print(
        f"[create_merged_df]: Dropped {len(page_views_df) - len(merged_df)} rows from page views"
        f"and {len(articles_df) - len(merged_df)} rows from articles"
    )

    # 1. Length
    merged_df["length"] = merged_df.encoded_text.map(
        lambda article: len(article.split(" "))
    )

    # 2. Images
    merged_df = await get_image_urls(merged_df)

    # 3. One-hot encode categories
    def one_hot_encode_categories(categories):
        return list(
            map(
                int,
                [
                    article_category in categories
                    for article_category in ARTICLE_CATEGORIES
                ],
            )
        )

    merged_df["categories_one_hot"] = merged_df.categories.map(
        one_hot_encode_categories
    )

    # 4. Calculate the number of days posted
    merged_df["n_days_posted"] = merged_df.date.map(lambda d: (datetime.now() - d).days)

    return merged_df


async def get_image_urls(merged_df):
    def get_image_url(html):
        soup = BeautifulSoup(html, "html.parser")
        images = soup.findAll("meta", {"property": "og:image"})
        if len(images) == 1:
            image = images[0]["content"]
            return image
        else:
            raise Exception("Can't find image")

    merged_df["image_url"] = None

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11"
        ),
    }

    print("[get_image-urls]: Retrieving image URLs")

    batch_size = 200
    for i in range(0, merged_df.shape[0], batch_size):
        print(
            f"[get_image-urls]: Processing batch {i // batch_size} of {merged_df.shape[0] // batch_size}"
        )
        subset = merged_df.iloc[i : i + batch_size]

        loop = asyncio.get_event_loop()
        requests = {
            "indices": [],
            "futures": [],
        }

        for j, row in subset.iterrows():
            if row.image_url is None:
                req = Request(row.link, headers=headers)
                future = loop.run_in_executor(None, urlopen, req)
                requests["indices"].append(j)
                requests["futures"].append(future)

        for j, res in zip(
            requests["indices"], await asyncio.gather(*requests["futures"])
        ):
            html = res.read()
            try:
                merged_df.loc[j, "image_url"] = get_image_url(html)
            except Exception:
                print(
                    f"[get_image-urls]: Failed to retrieve image URL for {merged_df.loc[j].link}"
                )

    return merged_df


async def main():
    args = parse_shell_args()

    if not args.force and os.path.isfile(args.output):
        print(
            f"Found existing CSV at {args.output}. To overwrite it, pass in the --force argument."
        )
        return

    with open(args.csv_one, "r") as page_views_csv_one, open(
        args.json_one, "r"
    ) as articles_json_one, open(args.csv_two, "r") as page_views_csv_two, open(
        args.json_two, "r"
    ) as articles_json_two:
        page_views_df_one = pd.read_csv(page_views_csv_one)
        articles_dict_one = json.load(articles_json_one)

        page_views_df_two = pd.read_csv(page_views_csv_two)
        articles_dict_two = json.load(articles_json_two)

        page_views_df_one = clean_page_views_df(page_views_df_one)
        articles_df_one = clean_articles_dict(articles_dict_one)

        page_views_df_two = clean_page_views_df(page_views_df_two)
        articles_df_two = clean_articles_dict(articles_dict_two)

        page_views_df = pd.concat([page_views_df_one, page_views_df_two])
        articles_df = pd.concat([articles_df_one, articles_df_two])

        merged_df = await create_merged_df(page_views_df, articles_df)

        print(f"[main]: Saving result to {args.output}")
        merged_df.to_csv(args.output)


if __name__ == "__main__":
    asyncio.run(main())
