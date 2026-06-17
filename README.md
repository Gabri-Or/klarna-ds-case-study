# Klarna Data Science Classification Case Study

This case consists of a supervised learning example, similar to what some Klarna teams are working with on a daily basis. Your task is to develop a model that predicts the probability of default on a purchase made with Klarna’s Pay Later payment method.

Pay Later is effectively a loan Klarna issues to the consumer, to finance their purchase at the point of sale. It works as follows:

- Consumers shop with Klarna, but pay for their purchase 14 days later. This allows consumers to see and feel, or try on the goods before having to pay for them.
- If the consumer has not paid Klarna after these 14 days, they will get a reminder and be given another 7 days to pay.
- If the customer has still not paid back after that period, they will enter debt collection and the debt is either sold off to debt collectors or eventually written off by Klarna. This is a “default”.

Your model will be used by the team responsible for Underwriting - this is the Klarna team who decides which loans to issue, based on the probability of default (and accept or decline transactions accordingly). This team expects to have accurate predictions and needs to have the flexibility to choose the level of credit risk Klarna takes.

To this end you are provided with a dataset that contains:

- The id of the loan
- The date the loan was issued
- The size of the loan
- The outstanding balance of the loan 14 and 21 days after it was issued.

Along with a number of features that you can use as predictors in your model. We use Python for modeling at Klarna so please use this programming language to craft your submission. A data dictionary is also supplied on the page below.

Your solution should contain:

- Code to host an API, which a reviewer should be able to host on their local system and make requests to.
- A 1-pager that summarizes how you chose a target definition, trained your model, and evaluated performance.
- Any code used to explore the data or develop the model, which a reviewer should be able to follow.

Our advice is to avoid spending too much time optimizing your prediction results. We are more interested in how you structure your solution, how you reason about the problem and how you validate your results. Showing off your skills in model building, analysis, and software engineering is more important than maximizing predictive performance. Good luck!

## Data Dictionary

| Feature | Definition |
| --- | --- |
| `loan_id` | A random, distinct ID for the loan |
| `loan_issue_date` | The date the loan was issued |
| `loan_amount` | The value of the loan being underwritten |
| `amount_outstanding_14d` | How much of the loan remained unpaid 14 days after the loan was issued |
| `amount_outstanding_21d` | How much of the loan remained unpaid 21 days after the loan was issued |
| `card_expiry_month` | The month the consumer’s payment card will expire |
| `card_expiry_year` | The year the consumer’s payment card will expire |
| `existing_klarna_debt` | How much the consumer already owed to Klarna at the time the loan was issued |
| `num_active_loans` | The number of loans the consumer needed to repay at the time the loan was issued |
| `days_since_first_loan` | How many days had passed since the consumer’s first loan, as of the time the current loan was issued |
| `new_exposure_7d` | How much Klarna had lent the consumer 7 days before the loan was issued |
| `new_exposure_14d` | How much Klarna had lent the consumer 14 days before the loan was issued |
| `num_confirmed_payments_3m` | How many repayments towards other loans the consumer had made 3 months before the loan was issued |
| `num_confirmed_payments_6m` | How many repayments towards other loans the consumer had made 6 months before the loan was issued |
| `num_failed_payments_3m` | How many repayments towards other loans the consumer had missed 3 months before the loan was issued |
| `num_failed_payments_6m` | How many repayments towards other loans the consumer had missed 6 months before the loan was issued |
| `num_failed_payments_1y` | How many repayments towards other loans the consumer had missed 1 year before the loan was issued |
| `amount_repaid_14d` | How much the consumer had repaid in the 14 days before the loan was issued |
| `amount_repaid_1m` | How much the consumer had repaid in the month before the loan was issued |
| `amount_repaid_3m` | How much the consumer had repaid in the 3 months before the loan was issued |
| `amount_repaid_6m` | How much the consumer had repaid in the 6 months before the loan was issued |
| `amount_repaid_1y` | How much the consumer had repaid the year before the loan was issued |
| `merchant_group` | These features describe the merchant where the consumer is shopping |
| `merchant_category` | These features describe the merchant where the consumer is shopping |
