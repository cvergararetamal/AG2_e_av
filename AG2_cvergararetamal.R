#!/usr/bin/env RScript
###############################################################
# Creador: Christian Vergara Retamal
# Usuario VIU: christian.vergararet
# Grado Master en Big Data y Ciencia de Datos
# Asignatura: Estadística Avanzada

# Versión 0.1
# Date: 25/11/2023

#####################Librerías##########################

library(skimr)
library(ggplot2)
library(readr)
library(dplyr)
library(glmnet)
library(pROC)
library(caret)
library(readr)
library(lubridate)

#####################Preparación & Análisis Exploratorio##########################

#Carga de datos
data_energia_es <- read_csv("spain_energy_market.csv")

#Primeros Registros
head(data_energia_es)
#Últimos Registros
tail(data_energia_es)

#Estadísticos básicos
summary(data_energia_es)

# Caracterización de las columnas
str(data_energia_es)
## Se verifica que la columna datetime efectivamente es de tipo date, con el objetivo de trabajar
## correctamente con series de tiempo

# Fráfica de la serie de tiempo de precio de energía en españa por fecha
ggplot(data_energia_es, aes(x =datetime,
                            y =value))+geom_line()+labs(title = "Precio de la energía en España",
                                                                x = "Fecha",y = "Precio")


# Observamos el comportamiento de la columna Precio, através de un histograma
# Histograma de los valores
ggplot(data_energia_es,aes(x = value)) +geom_histogram(bins = 30)+
  labs(title = "Histograma de los Precios de la Energía",x ="Precio",y = "Freq")



## Descomposición 
ts_data_energia_es <- ts(data_energia_es$value,frequency =365)#Datos dirios

#Utilizando Decompose
decomp_des_add <- decompose(ts_data_energia_es, type = "additive")
decomp_des_mult <- decompose(ts_data_energia_es, type = "multiplicative")


# Utilizando stl
decomp_stl <- stl(ts_data_energia_es, s.window="periodic", )


plot(decomp_des_add)
plot(decomp_des_mult)
plot(decomp_stl)



## Análisis Estacionariedad
valor_esperado <- mean(ts_data_energia_es)
varianza <- var(ts_data_energia_es)

print(valor_esperado)
print(varianza)


# Análisis en base a división de segmentos
size_ts <- length(ts_data_energia_es)
segment_length <- size_ts %/% 5 # Numero de segmentos, uno por año 2014 - 2018

segment_means <- 5
segment_variances <- 5
# Calcular las medias y varianzas de cada segmento
for (i in 1:5) {
  segment_data <- ts_data_energia_es[((i-1) * segment_length + 1):(i * segment_length)]
  segment_means[i] <-mean(segment_data)
  segment_variances[i] <-var(segment_data)
}

print(segment_means)
print(segment_variances)


# Análisis en base a ACF y PACF
acf(ts_data_energia_es, main="Autocorrelación")
pacf(ts_data_energia_es, main="Autocorrelación Parcial")



#####################Construcción y selección de modelos##########################

# se evaluaran algunos aspectos importantes a considerar,
# De acuerdo con el material visto en clases

# Diferenciacion
ts_data_energia_es_diff <- diff(ts_data_energia_es)
plot(ts_data_energia_es_diff, main = "Serie diff")
# Autocorrelación 
acf(ts_data_energia_es_diff, main = "ACF serie diff")
pacf(ts_data_energia_es_diff, main = "PACF serie diff")


# Se implementa un modelo arima para analizar los residuos
# Los parámetros de implementación corresponden a los siguientes dado el análisis previo
      # param 1 "p" -> Definido por la gráfic ade PACF, sugiriendo un modelo AR(1), por lo tanto p =1
      # param 2 "d" -> Grado de diferenciacion
      # param 3 "q" -> Definido por la gráfic ade ACF, comenzaremos con 1 para el modelo arima inicial
modelo_arima_1 <- arima(ts_data_energia_es, order = c(1, 1, 1))
residuos_modleo_arima_1 <- residuals(modelo_arima_1)

# Test ruido blanco
Box.test(residuos_modleo_arima_1, type = "Ljung-Box")
# ACF Residuoa
acf(residuos_modleo_arima_1, main = "Residuos arima 1")

summary(modelo_arima_1)
print(AIC(modelo_arima_1))


modelo_arima_2 <- arima(ts_data_energia_es, order = c(0, 1, 0))
residuos_modleo_arima_2 <- residuals(modelo_arima_2)

# Test ruido blanco
Box.test(residuos_modleo_arima_2, type = "Ljung-Box")
# ACF Residuoa
acf(residuos_modleo_arima_2, main = "Residuos arima 2")

summary(modelo_arima_2)
print(AIC(modelo_arima_2))



##### Prueba ajuste modelo SARIMA -> Componentes estacionarios
#modelo_sarima_1 <- arima(ts_data_energia_es, order = c(1, 1, 1),
#                         seasonal=list(order = c(1, 1, 1), period =365))

# -> El modelo no se ejecuta por las limitaciones de modelo arima: maximum lag is 350



### Dado lo anterior, y con el objetivo de encontrar y ajustar un mejor modelo. Se utilizará
# un periodo de estacionalidad mas acotado, por ejemplo en trimestres


modelo_sarima_1_trim <- arima(ts_data_energia_es, order =c(1,1,1),
                              seasonal =list(order=c(1, 1,1), period=4))
summary(modelo_sarima_1_trim)
print(AIC(modelo_sarima_1_trim))

# Residuos sarima 1

residuos_sarima_trim <- residuals(modelo_sarima_1_trim)
Box.test(residuos_sarima_trim, type="Ljung-Box")



####### Revisión modelo sarima
# -> Autocorrelacion significativa en los residuos implica ajustes

Box.test(residuos_sarima_trim, lag = 5, type = "Ljung-Box")
Box.test(residuos_sarima_trim, lag = 10, type = "Ljung-Box")
Box.test(residuos_sarima_trim, lag = 15, type = "Ljung-Box")
Box.test(residuos_sarima_trim, lag = 20, type = "Ljung-Box")



### Ajuste Iteración 2 Modelo SARIMA

# Se analizaran manualmente diferentes componenetes estacionales

## Periodo 12
modelo_sarima_2_param111_period12 <- arima(ts_data_energia_es, order = c(1, 1, 1),
                                   seasonal=list(order =c(1,1,1),period = 12))
print(summary(modelo_sarima_2_param111_period12))
print(AIC(modelo_sarima_2_param111_period12))

residuos_sarima_2_param111_period12 <- residuals(modelo_sarima_2_param111_period12)
Box.test(residuos_sarima_2_param111_period12, type = "Ljung-Box")


## Periodo 6
modelo_sarima_2_param111_period6 <- arima(ts_data_energia_es, order = c(1, 1, 1),
                                           seasonal=list(order =c(1,1,1),period = 6))
print(summary(modelo_sarima_2_param111_period6))
print(AIC(modelo_sarima_2_param111_period6))

residuos_sarima_2_param111_period6 <- residuals(modelo_sarima_2_param111_period6)
Box.test(residuos_sarima_2_param111_period6, type = "Ljung-Box")


### Ajuste Iteración 3 Modelo SARIMA

# Prueba valores diferenciación

## Periodo 12
modelo_sarima_2_param121_period12 <- arima(ts_data_energia_es, order = c(1, 2, 1),
                                           seasonal=list(order =c(1,1,1),period = 12))
print(summary(modelo_sarima_2_param121_period12))
print(AIC(modelo_sarima_2_param121_period12))

residuos_sarima_2_param121_period12 <- residuals(modelo_sarima_2_param121_period12)
Box.test(residuos_sarima_2_param121_period12, type = "Ljung-Box")


## Periodo 6
modelo_sarima_2_param121_period6 <- arima(ts_data_energia_es, order = c(1, 2, 1),
                                          seasonal=list(order =c(1,1,1),period = 6))
print(summary(modelo_sarima_2_param121_period6))
print(AIC(modelo_sarima_2_param121_period6))

residuos_sarima_2_param121_period6 <- residuals(modelo_sarima_2_param121_period6)
Box.test(residuos_sarima_2_param121_period6, type = "Ljung-Box")


## Periodo 4
modelo_sarima_2_param121_period4 <- arima(ts_data_energia_es, order = c(1, 2, 1),
                                          seasonal=list(order =c(1,1,1),period = 4))
print(summary(modelo_sarima_2_param121_period4))
print(AIC(modelo_sarima_2_param121_period4))

residuos_sarima_2_param121_period4 <- residuals(modelo_sarima_2_param121_period4)
Box.test(residuos_sarima_2_param121_period4, type = "Ljung-Box")



## Periodo 3
modelo_sarima_2_param121_period3 <- arima(ts_data_energia_es, order = c(1, 2, 1),
                                          seasonal=list(order =c(1,1,1),period = 3))
print(summary(modelo_sarima_2_param121_period3))
print(AIC(modelo_sarima_2_param121_period3))

residuos_sarima_2_param121_period3 <- residuals(modelo_sarima_2_param121_period3)
Box.test(residuos_sarima_2_param121_period3, type = "Ljung-Box")



### Ajuste Iteración 4 Modelo SARIMA

# Identificación valores atípicos 
ts_data_energia_es_aux <- ts_data_energia_es
media <- mean(ts_data_energia_es_aux)
desviacion_std <- sd(ts_data_energia_es_aux)
limites <- c(media - 3 *desviacion_std, media +3 *desviacion_std)

atipicos <- which(ts_data_energia_es_aux < limites[1] | ts_data_energia_es_aux > limites[2])

ts_data_energia_es_aux[atipicos] <- media #Reemplazocon la media

modelo_sarima_reajustado_copia <- arima(ts_data_energia_es_aux, order = c(1, 1, 1),
                                        seasonal =list(order = c(1, 1, 1), period = 12))
Box.test(residuals(modelo_sarima_reajustado_copia), type = "Ljung-Box")

## Mejora significativa! p-value = 0.000383

## Aplicaremos a los distintos sarimas realizados para sleccionar el mejor

print("IT4")

## Periodos 12
###111
modelo_sarima_3_param111_period12_sin_atipico <- arima(ts_data_energia_es_aux, order = c(1, 1, 1),
                                           seasonal=list(order =c(1,1,1),period = 12))
print(summary(modelo_sarima_3_param111_period12_sin_atipico))
print(AIC(modelo_sarima_3_param111_period12_sin_atipico))

residuos_sarima_3_param111_period12 <- residuals(modelo_sarima_3_param111_period12_sin_atipico)
Box.test(residuos_sarima_3_param111_period12, type = "Ljung-Box")

###121
modelo_sarima_3_param121_period12_sin_atipico <- arima(ts_data_energia_es_aux, order = c(1, 2, 1),
                                                       seasonal=list(order =c(1,1,1),period = 12))
print(summary(modelo_sarima_3_param121_period12_sin_atipico))
print(AIC(modelo_sarima_3_param121_period12_sin_atipico))

residuos_sarima_3_param121_period12 <- residuals(modelo_sarima_3_param121_period12_sin_atipico)
Box.test(residuos_sarima_3_param121_period12, type = "Ljung-Box")


#--------------#

## Periodos 6

###111
modelo_sarima_3_param111_period6_sin_atipico <- arima(ts_data_energia_es_aux, order = c(1, 1, 1),
                                                       seasonal=list(order =c(1,1,1),period = 6))
print(summary(modelo_sarima_3_param111_period6_sin_atipico))
print(AIC(modelo_sarima_3_param111_period6_sin_atipico))

residuos_sarima_3_param111_period6 <- residuals(modelo_sarima_3_param111_period6_sin_atipico)
Box.test(residuos_sarima_3_param111_period6, type = "Ljung-Box")

###121
modelo_sarima_3_param121_period6_sin_atipico <- arima(ts_data_energia_es_aux, order = c(1, 2, 1),
                                                       seasonal=list(order =c(1,1,1),period = 6))
print(summary(modelo_sarima_3_param121_period6_sin_atipico))
print(AIC(modelo_sarima_3_param121_period6_sin_atipico))

residuos_sarima_3_param121_period6 <- residuals(modelo_sarima_3_param121_period6_sin_atipico)
Box.test(residuos_sarima_3_param121_period6, type = "Ljung-Box")




#--------------#

## Periodo 4
###111
modelo_sarima_3_param111_period4_sin_atipico <- arima(ts_data_energia_es_aux, order = c(1, 1, 1),
                                                      seasonal=list(order =c(1,1,1),period = 4))
print(summary(modelo_sarima_3_param111_period4_sin_atipico))
print(AIC(modelo_sarima_3_param111_period4_sin_atipico))

residuos_sarima_3_param111_period4 <- residuals(modelo_sarima_3_param111_period4_sin_atipico)
Box.test(residuos_sarima_3_param111_period4, type = "Ljung-Box")

###121
modelo_sarima_3_param121_period4_sin_atipico <- arima(ts_data_energia_es_aux, order = c(1, 2, 1),
                                                      seasonal=list(order =c(1,1,1),period = 4))
print(summary(modelo_sarima_3_param121_period4_sin_atipico))
print(AIC(modelo_sarima_3_param121_period4_sin_atipico))

residuos_sarima_3_param121_period4 <- residuals(modelo_sarima_3_param121_period4_sin_atipico)
Box.test(residuos_sarima_3_param121_period4, type = "Ljung-Box")



#--------------#

## Periodo 3
###111
modelo_sarima_3_param111_period3_sin_atipico <- arima(ts_data_energia_es_aux, order = c(1, 1, 1),
                                                      seasonal=list(order =c(1,1,1),period = 3))
print(summary(modelo_sarima_3_param111_period3_sin_atipico))
print(AIC(modelo_sarima_3_param111_period3_sin_atipico))

residuos_sarima_3_param111_period3 <- residuals(modelo_sarima_3_param111_period3_sin_atipico)
Box.test(residuos_sarima_3_param111_period3, type = "Ljung-Box")

###121
modelo_sarima_3_param121_period3_sin_atipico <- arima(ts_data_energia_es_aux, order = c(1, 2, 1),
                                                      seasonal=list(order =c(1,1,1),period = 3))
print(summary(modelo_sarima_3_param121_period3_sin_atipico))
print(AIC(modelo_sarima_3_param121_period3_sin_atipico))

residuos_sarima_3_param121_period3 <- residuals(modelo_sarima_3_param121_period3_sin_atipico)
Box.test(residuos_sarima_3_param121_period3, type = "Ljung-Box")



### Ajuste Iteración 4 Modelo SARIMA: Fza Bruta
# -> Se mantendra el periodo 6, ya que fuel el con mejores resultados

# Modelo SARIMA con diferentes combinaciones de parámetros AR y MA y período estacional de 6
# Asumiendo que 'ts_data_energia_es_aux' es la serie temporal modificada sin atípicos

print("IT5")

modelo_sarima_111_111_6 <- arima(ts_data_energia_es_aux, order = c(1, 1, 1),
                                 seasonal = list(order = c(1, 1, 1), period = 6))
print(summary(modelo_sarima_111_111_6))
print(AIC(modelo_sarima_111_111_6))
Box.test(residuals(modelo_sarima_111_111_6), type = "Ljung-Box")

modelo_sarima_121_111_6 <- arima(ts_data_energia_es_aux,
                                 order = c(1, 2, 1), seasonal = list(order = c(1, 1, 1), period = 6))
print(summary(modelo_sarima_121_111_6))
print(AIC(modelo_sarima_121_111_6))
Box.test(residuals(modelo_sarima_121_111_6), type = "Ljung-Box")

modelo_sarima_112_111_6 <- arima(ts_data_energia_es_aux,
                                 order = c(1, 1, 2), seasonal = list(order = c(1, 1, 1), period = 6))
print(summary(modelo_sarima_112_111_6))
print(AIC(modelo_sarima_112_111_6))
Box.test(residuals(modelo_sarima_112_111_6), type = "Ljung-Box")

modelo_sarima_211_111_6 <- arima(ts_data_energia_es_aux, order = c(2, 1, 1),
                                 seasonal = list(order = c(1, 1, 1), period = 6))
print(summary(modelo_sarima_211_111_6))
print(AIC(modelo_sarima_211_111_6))
Box.test(residuals(modelo_sarima_211_111_6), type = "Ljung-Box")

modelo_sarima_212_111_6 <- arima(ts_data_energia_es_aux, order = c(2, 1, 2),
                                 seasonal = list(order = c(1, 1, 1), period = 6))
print(summary(modelo_sarima_212_111_6))
print(AIC(modelo_sarima_212_111_6))
Box.test(residuals(modelo_sarima_212_111_6), type = "Ljung-Box")



modelo_sarima_221_111_6 <- arima(ts_data_energia_es_aux, order = c(2, 2, 1),
                                 seasonal = list(order = c(1, 1, 1), period = 6))
print(summary(modelo_sarima_221_111_6))
print(AIC(modelo_sarima_221_111_6))
Box.test(residuals(modelo_sarima_221_111_6), type = "Ljung-Box")



#### Selección mejor modelo
summary(modelo_sarima_112_111_6)

# Análisis de residuos
residuos_best_model <- residuals(modelo_sarima_112_111_6)
plot(residuos_best_model, main="Residuos del Modelo modelo_sarima_112_111_6")
acf(residuos_best_model, main="ACF de los Residuos")
acf(residuos_best_model, main="PACF de los Residuos")



## Pronóstico
# Cargar la librería forecast si aún no está cargada
library(forecast)
pronosticos <- forecast(modelo_sarima_112_111_6, h=12) 
plot(pronosticos, main="Pronósticos a Futuro")

plot(ts_data_energia_es_aux, main="Serie Temporal Original y Ajuste del Modelo")
lines(fitted(modelo_sarima_112_111_6), col="red")

# Gráfico de pronósticos futuros
plot(pronosticos, main="Pronósticos Futuros con el Modelo SARIMA(1,1,2)x(1,1,1)[6]")


