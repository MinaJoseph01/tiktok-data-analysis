# tiktok-data-analysis
 
**Repository Name:** tiktok-data-analysis

**Description:**

This repository contains the code and analysis for an exploratory data analysis project on a TikTok dataset. The goal of this project is to understand trends and engagement metrics related to TikTok videos and ultimately use machine learning to classify videos as claims or opinions.

**Project Structure:**

1. **Imports and Data Loading:**
   - Imported necessary libraries: `pandas` and `numpy`.
   - Loaded the dataset into a DataFrame.

   ```python
   import pandas as pd
   import numpy as np

   # Load dataset into dataframe
   data = pd.read_csv("tiktok_dataset.csv")
   data.head()
   ```

2. **Initial Data Inspection:**
   - Displayed the first ten rows of the DataFrame.
   - Obtained summary information using `data.info()` and `data.describe()`.

   ```python
   # Display the first ten rows
   data.head(10)
   
   # Get summary info
   data.info()
   data.describe()
   ```

3. **Understanding Variables:**
   - Examined the `claim_status` variable and its distribution.

   ```python
   claim_status = data['claim_status'].value_counts()
   print(claim_status)
   ```

4. **Engagement Analysis:**
   - Calculated mean and median view counts for claims and opinions.
   - Investigated trends in author ban statuses.

   ```python
   claims = data[data['claim_status'] == 'claim']
   print(claims['video_view_count'].mean())
   print(claims['video_view_count'].median())

   opinion = data[data['claim_status'] == 'opinion']
   print(opinion['video_view_count'].mean())
   print(opinion['video_view_count'].median())

   data.groupby(['claim_status', 'author_ban_status']).count()[['#']]
   ```

5. **Engagement Rates Calculation:**
   - Created new columns for likes per view, comments per view, and shares per view.

   ```python
   # Create new columns
   data['likes_per_view'] = data['video_like_count'] / data['video_view_count']
   data['comments_per_view'] = data['video_comment_count'] / data['video_view_count']
   data['shares_per_view'] = data['video_share_count'] / data['video_view_count']
   ```

6. **Aggregated Analysis:**
   - Compiled engagement rates for different combinations of claim status and author ban status.

   ```python
   data.groupby(['claim_status', 'author_ban_status']).agg(
       {'likes_per_view': ['count', 'mean', 'median'],
        'comments_per_view': ['count', 'mean', 'median'],
        'shares_per_view': ['count', 'mean', 'median']})
   ```

**Results and Insights:**
- Videos with claim status generally have higher engagement compared to opinion videos.
- Banned authors tend to have higher engagement rates, suggesting controversial content might attract more interactions.

**Conclusions:**
This analysis provides a comprehensive understanding of engagement metrics on TikTok videos, laying the foundation for machine learning classification tasks.

