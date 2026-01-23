#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
library(tidyverse)
library(tidymodels)
library(skimr)
library(MLearnYRBSS)
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#| echo: true
#| code-line-numbers: "1|3|4|5|6|7|9|10|11|12|13"

data("riskyBehaviors")

riskyBehaviors_analysis <-
    riskyBehaviors |>
    mutate(
        UsedAlcohol = case_when(
            AgeFirstAlcohol == 1 ~ 0,
            AgeFirstAlcohol %in% c(2, 3, 5, 6, 4, 7) ~ 1,
            TRUE ~ NA
        )
    ) |>
    mutate(UsedAlcohol = factor(UsedAlcohol)) |>
    drop_na(UsedAlcohol) |>
    select(
        -c(
            AgeFirstAlcohol,
            DaysAlcohol,
            BingeDrinking,
            LargestNumberOfDrinks,
            SourceAlcohol,
            SourceAlcohol
        )
    )

#
#
#
#
#
#
#| echo: true
#| code-line-numbers: "1|3|4|6|7|9"
#| output-location: fragment

set.seed(2023)

alcohol_split <- initial_split(riskyBehaviors_analysis, strata = UsedAlcohol)

alcohol_train <- training(alcohol_split)
alcohol_test <- testing(alcohol_split)

alcohol_split
#
#
#
#
#
#
library(janitor)
#
#
#
#
#
#
#
#| echo: true
#| code-line-numbers: "1|2|3|4"
#| output-location: fragment

alcohol_train |>
    tabyl(UsedAlcohol) |>
    adorn_pct_formatting(0) |>
    adorn_totals()


#
#
#
#
#
#
#
#| echo: true
#| code-line-numbers: "1|2|3|4"
#| output-location: fragment

alcohol_test |>
    tabyl(UsedAlcohol) |>
    adorn_pct_formatting(0) |>
    adorn_totals()
#
#
#
#
#
#
#
#
#| echo: true
#| code-line-numbers: "1|3|4|5"
#| output-location: fragment

set.seed(2023)

cv_alcohol <- rsample::vfold_cv(alcohol_train, strata = UsedAlcohol)
cv_alcohol
#
#
#
#
#
#
#
#| echo: true
#| code-line-numbers: "1|2|3|4"
#| output-location: fragment

alcohol_recipe <-
    recipe(formula = UsedAlcohol ~ ., data = alcohol_train) |>
    step_impute_mode(all_nominal_predictors()) |>
    step_impute_mean(all_numeric_predictors())


#
#
#
#
#
#| echo: true
#| code-line-numbers: "1|2|3|4|5|6|7|9"
#| output-location: fragment

cart_spec <-
    decision_tree(
        cost_complexity = tune(),
        tree_depth = tune(),
        min_n = tune()
    ) |>
    set_engine("rpart") |>
    set_mode("classification")

cart_spec
#
#
#
#
#
#
#| echo: true
#| code-line-numbers: "1|2|3|4|6"
#| output-location: fragment

cart_workflow <-
    workflow() |>
    add_recipe(alcohol_recipe) |>
    add_model(cart_spec)

cart_workflow
#
#
#
#
#
#| echo: true
#| code-line-numbers: "1|2|3|4|5|6"
#| output-location: fragment

tree_grid <-
    grid_regular(cost_complexity(), tree_depth(c(2, 5)), min_n(), levels = 4)
tree_grid

#
#
#
#
#
#| echo: true
#| eval: false
#| code-line-numbers: "1|3|5|6|7|8|10"
#| output-location: fragment

doParallel::registerDoParallel()

cart_tune <-
    cart_workflow %>%
    tune_grid(
        resamples = cv_alcohol,
        grid = tree_grid,
        metrics = metric_set(roc_auc),
        control = control_grid(save_pred = TRUE)
    )

doParallel::stopImplicitCluster()
#
#
#
#
#| echo: false

# saveRDS(cart_tune, "outputs/cart_tune.rds")
cart_tune <- readRDS("outputs/cart_tune.rds")
#
#
#
#
#
#| echo: true
#| code-line-numbers: "1"
#| output-location: fragment

show_best(cart_tune, metric = "roc_auc")
#
#
#
#
#
#| echo: true
#| code-line-numbers: "1|2|3|5"
#| output-location: fragment

bestPlot_cart <-
    autoplot(cart_tune)

bestPlot_cart
#
#
#
#
#
#| echo: true
#| code-line-numbers: "1|2|3|4"
#| output-location: fragment

best_cart <- select_best(
    cart_tune,
    metric = "roc_auc"
)

best_cart
#
#
#
#
#
#| echo: true
#| code-line-numbers: "1|2"
#| output-location: fragment

cart_final_wf <- finalize_workflow(cart_workflow, best_cart)
cart_final_wf
#
#
#
#
#
#
#
#
#
#| echo: true
#| code-line-numbers: "1|2|3|5"
#| output-location: fragment

cart_fit <- fit(
    cart_final_wf,
    alcohol_train
)

cart_fit
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#| echo: true
#| code-line-numbers: "1|2|3|5"
#| output-location: fragment

tree_pred <-
    augment(cart_fit, alcohol_train) |>
    select(UsedAlcohol, .pred_class, .pred_1, .pred_0)

tree_pred

#
#
#
#
#
#
#
#| echo: true
#| code-line-numbers: "1|2|3|4|5|6|8"
#| output-location: fragment

roc_tree <-
    tree_pred |>
    roc_curve(truth = UsedAlcohol, .pred_1, event_level = "second") |>
    autoplot()

roc_tree

#
#
#
#
#
#
#| echo: true
#| code-line-numbers: "1|2|3|4"
#| output-location: fragment

tree_pred |> 
  roc_auc(truth = UsedAlcohol, 
           .pred_1, 
           event_level = "second")

#
#
#
#
#
#
#
#
#| echo: true
#| code-line-numbers: "1|2|3"
#| output-location: fragment
fit_resamples(cart_final_wf, resamples = cv_alcohol) |>
    collect_metrics()
#
#
#
#
#
#| echo: true
#| code-line-numbers: "1|2|3"
#| output-location: fragment

cart_fit |>
    extract_fit_engine() |>
    rpart.plot::rpart.plot(roundint = FALSE)
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
