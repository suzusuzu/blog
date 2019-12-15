library(fable)
library(tsibble)
library(lubridate)
library(dplyr)

AirPassengers %>%
  as_tsibble %>%
  model(
    ets = ETS(value),
    ets_log = ETS(log(value)),
    ets_box_cox = ETS(box_cox(value, 0.3)),
    arima = ARIMA(value),
    arima_log = ARIMA(log(value)),
    arima_box_cox = ARIMA(box_cox(value, 0.3)),
    snaive = SNAIVE(value),
    snaive_log = SNAIVE(log(value)),
    snaive_box_cox = SNAIVE(box_cox(value, 0.3)),
    nnetar = NNETAR(value),
    nnetar_log = NNETAR(log(value)),
    nnetar_box_cox = NNETAR(box_cox(value, 0.3))
  ) %>%
  forecast(h = "2 years") %>% 
  autoplot(filter(as_tsibble(AirPassengers), year(index) > 1960), level = NULL)
