library(ggplot2)
library(kableExtra)
codenames <- read.csv("/Users/nicholaspatrick/Desktop/Game/codenames/results/codenames_results.csv", 
                      col.names = c("game",
                                    "score",
                                    "red.words",
                                    "blue.words",
                                    "civilians",
                                    "assassin",
                                    "codemaster",
                                    "guesser",
                                    "seed",
                                    "time"),
                      header = F)

codenames$codemaster <- factor(codenames$codemaster,
                               levels = c("W2V and WordNet",
                                                "W2V 03",
                                                "W2V 05",
                                                "W2V 07",
                                                "WordNet Lin"))


codenames |> 
  group_by(codemaster, guesser) |> 
  summarise(avg.score = mean(score),
            sd.score = sd(score))


# table ------------------------------------------------------------------
result <- df %>%
  group_by(var1) %>%
  mutate(var2 = replace(var2, duplicated(var2), '')) %>%
  ungroup %>%
  mutate(var1 = replace(var1, duplicated(var1), ''))

table <- codenames |> 
  group_by(codemaster, guesser) |>
  summarize(avg.score = round(mean(score),2),
            sd.score = round(sd(score),2),
            games = n(),
            num.assassin = sum(assassin)) |> 
  ungroup() |> 
  mutate(codemaster = as.character(codemaster),
         codemaster = replace(codemaster, duplicated(codemaster), ''))

table |> 
  kbl(col.names = c("Codemaster",
                    "Guesser",
                    "Mean Score",
                    "SD",
                    "Number of Games",
                    "Number of Times Assassin Was Guessed"),
      caption = "Results of Test Games") |> 
  kable_classic_2(full_width = T, html_font = "Cambria",
                  lightable_options = "striped")
  


# boxplot -----------------------------------------------------------------
codenames |> 
ggplot( aes(x = codemaster, y = score)) +
  geom_boxplot() +
  stat_summary(geom = "errorbar",
               fun.min = mean,
               fun = mean,
               fun.max = mean,
               width = .75, 
               col = "blue") +
  labs(x = "Codemaster",
       y = "Score",
       title = "Boxplot of Scores By Codemaster") +
  geom_jitter(width = 0.2,
              height = 0) +
  theme_classic()

# line plot

codenames |> 
  filter(score != 25) |> 
  mutate(players = paste(codemaster, guesser, sep = ", ")) |> 
  ggplot(aes(x = seed, y = score, color = players)) +
  geom_point() + 
  geom_line() +
  scale_color_manual(values = c("red2","red2",
                                "blue4","blue4",
                                "grey3","grey3",
                                "yellow3","yellow3",
                                "lightblue", "lightblue"))
  

# Histogram

codenames |> 
  mutate(players = paste(codemaster, guesser, sep = ", ")) |> 
  ggplot(aes(x = score)) +
  geom_histogram() + facet_wrap(~players)
  