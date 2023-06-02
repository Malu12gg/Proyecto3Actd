#(recordar instalar paquetes si no se tienen)
install.packages("readxl")
install.packages("dplyr")
install.packages("ggplot2")
install.packages("GGally")
install.packages("Hmisc")
install.packages("corrplot")
install.packages("PerformanceAnalytics")
install.packages("vcd")
#Cargar paquetes
library(readxl)
library(dplyr)
library(ggplot2)
library(GGally)
library(Hmisc)
library(corrplot)
library(PerformanceAnalytics)
library(vcd)
install.packages("vcd")
library(vcd)
library(discretization)
#importar base de datos (recordar cambiar directorio)
base <-read.csv(file = "C:\\Users\\ricky\\Downloads\\Subconjunto.csv", header= TRUE,sep =",")
round(cor(base),2)
Subconjunto <- read.csv("C:/Users/ricky/Downloads/Subconjunto.csv")
subset_data <- Subconjunto[Subconjunto$cole_depto_ubicacion %in% c("BOGOTÃ", "ANTIOQUIA"), ] 
subset_data2 <- subset_data[subset_data$periodo %in% c("20222", "20224", "20221"), ]
subset_data3 <- subset_data2[subset_data2$cole_calendario %in% c("A"), ] 
subset_data4<- subset_data3[subset_data3$cole_jornada %in% c("MAÃ‘ANA", "NOCHE", "TARDE"), ] 

subset_data4$cole_area_ubicacion
filtered_data <- subset(subset_data3, select = c(cole_depto_ubicacion, cole_cod_mcpio_ubicacion, cole_area_ubicacion, cole_caracter,cole_naturaleza, cole_jornada, cole_genero,fami_estratovivienda, punt_global),)

write.csv(filtered_data, "datosREd.csv")
 tabla <-table(filtered_data$cole_depto_ubicacion, filtered_data$cole_cod_mcpio_ubicacion )


# Calcular el coeficiente de correlación de Cramer

cramer_v <- function(x, y) {
  contingency_table <- table(x, y)
  chi_square <- chisq.test(contingency_table)$statistic
  n <- sum(contingency_table)
  sqrt(chi_square / (n * min(dim(contingency_table)) - 1))
} 
resultado <-cramer_v(Subconjunto$cole_caracter,Subconjunto$fami_estratovivienda) 
resultado

