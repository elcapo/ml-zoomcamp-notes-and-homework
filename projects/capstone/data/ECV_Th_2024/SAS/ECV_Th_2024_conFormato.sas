/**********************************************************************************************************************					
Instituto Nacional de Estadï¿½stica (INE) www.ine.es					
***********************************************************************************************************************					
					
DESCRIPCIï¿½N:					
Este programa genera un fichero SAS con formatos, partiendo de un fichero sin ellos.					
					
Consta de las siguientes partes:					
	* 1. Definir la librerï¿½a de trabajo --> Libname				
	* 2. Definiciï¿½n de formatos --> PROC FORMAT				
	* 3. Vincular formatos a la base de datos --> PASO data				
					
 Entrada:                                                           					
     - Fichero SAS sin formatos: 	 ECV_Th_2023.sas7bdat				
 Salida:                                                           					
     - Fichero SAS con formatos: 	 ECV_Th_2023_conFormato.sas7bdat				
					
Donde:					
	* Operaciï¿½n: Encuesta de Condiciones de Vida
    * Fichero de datos bï¿½sicos del hogar_Fichero P(ECV_Tp)
	* Periodo: 2023
					
************************************************************************************************************************/					
		
/* Directorio de trabajo para la operaciï¿½n estadï¿½stica */
*%let siglas_periodo = ECV_Tp_2023;
*%let conFormato = _conFormato;
					
/*1) Definir la librerï¿½a de trabajo: introducir el directorio que desee como librerï¿½a					
(se da como ejemplo 'C:\Mis resultados'), y copiar en ese directorio el fichero sas "ECV_Tp_2022.sas7bdat"*/					
					
*libname ROutput 'C:\Mis resultados';	

options fmtsearch = (ROutput ROutput.cat1);

* 2) DEFINICIï¿½N DE FORMATOS;
PROC FORMAT LIBRARY=ROutput.cat1;
*Tablas1.......................;
value $TH120S

"1"="Con mucha dificultad"
"2"="Con dificultad"
"3"="Con cierta dificultad"
"4"="Con cierta facilidad"
"5"="Con facilidad"
"6"="Con mucha facilidad"
;
value $THCARGS

"1"="Una carga pesada"
"2"="Una carga razonable"
"3"="Ninguna carga"
;
value $TH010H

"1"="Vivienda unifamiliar independiente"
"2"="Vivienda unifamiliar adosada o pareada"
"3"="Piso o apartamento en un edificio con menos de 10 viviendas"
"4"="Piso o apartamento en un edificio con 10 viviendas o más"
;
value $TH021H

"1"="En propiedad sin hipoteca"
"2"="En propiedad con hipoteca"
"3"="En alquiler o realquiler a precio de mercado"
"4"="En alquiler o realquiler a precio inferior al de mercado"
"5"="En cesion gratuita"
;
value $TH060X

"  "="No consta"
"1"="Una persona: hombre de menos de 30 años"
"2"="Una persona: hombre de entre 30 y 64 años"
"3"="Una persona: hombre de 65 o más años"
"4"="Una persona: mujer de menos de 30 años"
"5"="Una persona: mujer de entre 30 y 64 años"
"6"="Una persona: mujer de 65 o más años"
"7"="2 adultos sin niños dependientes económicamente, al menos una persona de 65 o más años"
"8"="2 adultos sin niños dependientes económicamente, teniendo ambos menos de 65 años"
"9"="Otros hogares sin niños dependientes económicamente"
"10"="Un adulto con al menos un niño dependiente"
"11"="Dos adultos con un niño dependiente"
"12"="Dos adultos con dos niños dependientes"
"13"="Dos adultos con tres o más niños dependientes"
"14"="Otros hogares con niños dependientes"
;


*Tablas2.......................;
value $T_Flag

"-6"="El número de horas varía (incluso no es posible calcular una media para 4 semanas) [PL060_F]"
"-5"="No aplicable (no convivía con el progenitor)"
"-4"="No aplicable"
"-2"="No aplicable (según apartado)"
"-1"="No consta"
"1"="Variable completada"
;
value $T_SiNo

"1"="Sí"
"2"="No"
"9"="No aplicable"
;
value $T_Si2No

"1"="Sí"
"2"="No (por no poder permitírselo)"
"3"="No (otro motivo)"
;
value $TP040H

"1"="Sí, al menos en una ocasión"
"2"="No, en ninguna ocasión"
;
value $TP016L

"1"="No"
"2"="Sí, pero solo de forma ocasional"
"3"="Sí, de forma regular"
;

*Tablas2.......................;
value $T_Flag

"-2"="No aplicable (según apartado)"
"-1"="No consta"
"1"="Variable completada"
;
value $T_00Flg

"99"="Indicadores renta: T_00Flg"
;
value $T_2SiNo

"1"="Sí, solamente una vez"
"2"="Sí, dos veces o más"
"3"="No"
;
value $T_Si2No

"1"="Sí"
"2"="No (por no poder permitírselo)"
"3"="No (otro motivo)"
;
value $T_SiNo

"0"="No (en vhPobreza, vhMATDEP)"
"1"="Sí"
"2"="No"
;
value $TH080D

"1"="Sí"
"2"="No, el hogar no puede permitírselo"
"3"="No, por otras razones"
;
value $TH010I

"1"="Sí, los ingresos han aumentado"
"2"="No, los ingresos del hogar se han mantenido más o menos igual"
"3"="Sí, los ingresos han disminuido"
;
value $TH020I

"1"="Revalorización anual del salario"
"2"="Aumento de las horas trabajadas o del salario del trabajo actual"
"3"="Incorporación al trabajo después de una ausencia por enfermedad, maternidad/paternidad, cuidado de niños o cuidado de enfermos o mayores"
"4"="Comienzo o cambio de trabajo"
"5"="Cambios en la composición del hogar"
"6"="Percepción o incremento de las prestaciones sociales"
"7"="Otros motivos"
;
value $TH030I

"1"="Reducción de las horas trabajadas o del salario del trabajo actual"
"2"="Maternidad/paternidad, cuidado de niños o cuidado de enfermos o mayores"
"3"="Cambio de trabajo"
"4"="Pérdida de trabajo/desempleo"
"5"="Imposibilidad para trabajar por enfermedad o incapacidad"
"6"="Divorcio, separación u otros cambios en la composición del hogar"
"7"="Jubilación"
"8"="Eliminación o reducción de las prestaciones sociales"
"9"="Otros motivos"
;
value $TH040I

"1"="Mejorar"
"2"="Mantenerse más o menos igual"
"3"="Empeorar"
;

*Tablas3.......................;
value TH030H

10="No de habitaciones (>10)"
;

*TablasMI.......................;
value $TH010CH

"1"="Sí, al menos en una ocasión"
"2"="No, en ninguna ocasión"
;
value $TMI_1F

"-4"="No aplicable (no hay menores de 16 años)"
"-2"="No aplicable (ninguno de los niños del hogar ha necesitado realmente asistencia médica)"
"-1"="No consta"
"1"="Variable completada"
;
value $TH020CH

"1"="No se lo podía permitir"
"2"="Estaba en una lista de espera o no tenía volante"
"3"="No disponía de tiempo debido al trabajo, al cuidado de niños o de otras personas"
"4"="Demasiado lejos para viajar o sin medios de transporte"
"5"="Otras razones"
;
value $TMI_2F

"-4"="No aplicable (no hay menores de 16 años)"
"-2"="No aplicable (HCH010 distinto de 1)"
"-1"="No consta"
"1"="Variable completada"
;
value $TMI_3F

"-4"="No aplicable (no hay menores de 16 años)"
"-2"="No aplicable (ninguno de los niños ha necesitado realmente asistencia dental)"
"-1"="No consta"
"1"="Variable completada"
;
value $TMI_4F

"-4"="No aplicable (no hay menores de 16 años)"
"-2"="No aplicable (HCH030 distinto de 1)"
"-1"="No consta"
"1"="Variable completada"
;
value $TH100D

"1"="Sí"
"2"="No, el hogar no puede permitírselo"
"3"="No, por otras razones"
;
value $TMI_5F

"-4"="No aplicable (no hay menores de 16 años)"
"-1"="No consta"
"1"="Variable completada"
;
value $TMI_6F

"-4"="No aplicable (no hay menores de 16 años)"
"-2"="No aplicable (ningún menor asiste a la escuela)"
"-1"="No consta"
"1"="Variable completada"
;

*TablasMA.......................;
value $TH040C

"1"="Con mucha dificultad"
"2"="Con dificultad"
"3"="Con alguna dificultad"
"4"="Con cierta facilidad"
"5"="Con facilidad"
"6"="Con mucha facilidad"
;
value $TMA_1F

"-4"="No aplicable (no se utilizan servicios de cuidado de niños o no se paga por esos servicios o ningún de los niños asiste a preescolar ni a la escuela)"
"-2"="No aplicable (no hay menores de 13 años)"
"-1"="No consta"
"1"="Variable completada"
;
value $TMA_2F

"-4"="No aplicable (no se utilizan servicios de cuidado de niños o no se paga por esos servicios o ningún de los niños asiste a la guardería ni a preescolar ni a la escuela)"
"-2"="No aplicable (no hay menores de 13 años)"
"-1"="No consta"
"1"="Variable completada"
;
value $TH190C

"1"="Sí"
"2"="No"
;
value $TMA_3F

"-1"="No consta"
"1"="Variable completada"
;
value $TMA_4F

"-2"="No aplicable (no hay nadie en el hogar que necesite ayuda (HC190 = 2))"
"-1"="No consta"
"1"="Variable completada"
;
value $TH221C

"1"="Pagado íntegramente por la Administración Pública u otra institución"
"2"="El hogar paga una parte del coste ya que está subvencionado o recibe una ayuda"
"3"="El hogar paga el coste íntegro"
"4"="No sabe"
;
value $TMA_5F

"-2"="No aplicable (no se utilizan servicios de cuidados a domicilio (HC200 = 2) o no hay nadie en el hogar que necesite ayuda (HC190 = 2))"
"-1"="No consta"
"1"="Variable completada"
;
value $TMA_6F

"-2"="No aplicable (no se utilizan servicios de cuidados a domicilio (HC200 = 2) o no hay nadie en el hogar que necesite ayuda (HC190 = 2) o no se paga por esos servicios (HC221 = 1))"
"-1"="No consta"
"1"="Variable completada"
;
value $TH250C

"1"="El hogar no puede permitírselo"
"2"="La persona que los necesita los rechaza"
"3"="No están disponibles esos servicios"
"4"="La calidad de los servicios disponibles no es satisfactoria"
"5"="Otras razones"
;
value $TMA_7F

"-2"="No aplicable (no hay nadie en el hogar que necesite ayuda (HC190 = 2) o ningún miembro del hogar necesita o necesita más cuidados a domicilio (HC240 = 2))"
"-1"="No consta"
"1"="Variable completada"
;
value $TH300C

"1"="Una carga pesada"
"2"="Una carga razonable"
"3"="Ninguna carga"
"4"="Ningún miembro del hogar usa el transporte público"
;




* 3) VINCULAR FORMATOS A LA BASE DE DATOS;

	DATA ROutput.&siglas_periodo.&conFormato;
		set ROutput.&siglas_periodo;






FORMAT HB050_F $T_Flag.;

FORMAT HB060_F $T_Flag.;

FORMAT HB070_F $T_Flag.;

FORMAT HB080_F $T_Flag.;

FORMAT HB100_F $T_Flag.;

FORMAT HB120_F $T_Flag.;

FORMAT HY020_F $T_00Flg.;

FORMAT HY022_F $T_00Flg.;

FORMAT HY023_F $T_00Flg.;

FORMAT HY030N_F $T_00Flg.;

FORMAT HY040N_F $T_00Flg.;

FORMAT HY050N_F $T_00Flg.;

FORMAT HY060N_F $T_00Flg.;

FORMAT HY070N_F $T_00Flg.;

FORMAT HY080N_F $T_00Flg.;

FORMAT HY081N_F $T_00Flg.;

FORMAT HY090N_F $T_00Flg.;

FORMAT HY100N_F $T_00Flg.;

FORMAT HY110N_F $T_00Flg.;

FORMAT HY120N_F $T_00Flg.;

FORMAT HY130N_F $T_00Flg.;

FORMAT HY131N_F $T_00Flg.;

FORMAT HY145N_F $T_00Flg.;

FORMAT HY170N_F $T_00Flg.;

FORMAT HY010_F $T_00Flg.;

FORMAT HY040G_F $T_00Flg.;

FORMAT HY050G_F $T_00Flg.;

FORMAT HY060G_F $T_00Flg.;

FORMAT HY070G_F $T_00Flg.;

FORMAT HY080G_F $T_00Flg.;

FORMAT HY081G_F $T_00Flg.;

FORMAT HY090G_F $T_00Flg.;

FORMAT HY100G_F $T_00Flg.;

FORMAT HY110G_F $T_00Flg.;

FORMAT HY120G_F $T_00Flg.;

FORMAT HY130G_F $T_00Flg.;

FORMAT HY131G_F $T_00Flg.;

FORMAT HY140G_F $T_00Flg.;
FORMAT HS011 $T_2SiNo.;
FORMAT HS011_F $T_Flag.;
FORMAT HS021 $T_2SiNo.;
FORMAT HS021_F $T_Flag.;
FORMAT HS022 $T_SiNo.;
FORMAT HS022_F $T_Flag.;
FORMAT HS031 $T_2SiNo.;
FORMAT HS031_F $T_Flag.;
FORMAT HS040 $T_SiNo.;
FORMAT HS040_F $T_Flag.;
FORMAT HS050 $T_SiNo.;
FORMAT HS050_F $T_Flag.;
FORMAT HS060 $T_SiNo.;
FORMAT HS060_F $T_Flag.;
FORMAT HS090 $T_Si2No.;
FORMAT HS090_F $T_Flag.;
FORMAT HS110 $T_Si2No.;
FORMAT HS110_F $T_Flag.;
FORMAT HS120 $TH120S.;
FORMAT HS120_F $T_Flag.;
FORMAT HS150 $THCARGS.;
FORMAT HS150_F $T_Flag.;
FORMAT HD080 $TH080D.;
FORMAT HD080_F $T_Flag.;
FORMAT HH010 $TH010H.;
FORMAT HH010_F $T_Flag.;
FORMAT HH021 $TH021H.;
FORMAT HH021_F $T_Flag.;
FORMAT HH030 TH030H.;
FORMAT HH030_F $T_Flag.;
FORMAT HH050 $T_SiNo.;
FORMAT HH050_F $T_Flag.;

FORMAT HH060_F $T_Flag.;

FORMAT HH070_F $T_Flag.;
FORMAT HI010 $TH010I.;
FORMAT HI010_F $T_Flag.;
FORMAT HI020 $TH020I.;
FORMAT HI020_F $T_Flag.;
FORMAT HI030 $TH030I.;
FORMAT HI030_F $T_Flag.;
FORMAT HI040 $TH040I.;
FORMAT HI040_F $T_Flag.;

FORMAT cuotahip_F $T_Flag.;

FORMAT HX060 $TH060X.;



FORMAT vhPobreza $T_SiNo.;
FORMAT vhMATDEP $T_SiNo.;
FORMAT HCH010 $TH010CH.;
FORMAT HCH010_F $TMI_1F.;
FORMAT HCH020 $TH020CH.;
FORMAT HCH020_F $TMI_2F.;
FORMAT HCH030 $TH010CH.;
FORMAT HCH030_F $TMI_3F.;
FORMAT HCH040 $TH020CH.;
FORMAT HCH040_F $TMI_4F.;
FORMAT HD100 $TH100D.;
FORMAT HD100_F $TMI_5F.;
FORMAT HD110 $TH100D.;
FORMAT HD110_F $TMI_5F.;
FORMAT HD120 $TH100D.;
FORMAT HD120_F $TMI_5F.;
FORMAT HD140 $TH100D.;
FORMAT HD140_F $TMI_5F.;
FORMAT HD150 $TH100D.;
FORMAT HD150_F $TMI_5F.;
FORMAT HD160 $TH100D.;
FORMAT HD160_F $TMI_5F.;
FORMAT HD170 $TH100D.;
FORMAT HD170_F $TMI_5F.;
FORMAT HD180 $TH100D.;
FORMAT HD180_F $TMI_5F.;
FORMAT HD190 $TH100D.;
FORMAT HD190_F $TMI_5F.;
FORMAT HD200 $TH100D.;
FORMAT HD200_F $TMI_5F.;
FORMAT HD210 $TH100D.;
FORMAT HD210_F $TMI_6F.;
FORMAT HD220 $TH100D.;
FORMAT HD220_F $TMI_6F.;
FORMAT HD240 $TH100D.;
FORMAT HD240_F $TMI_5F.;
FORMAT HC040 $TH040C.;
FORMAT HC040_F $TMA_1F.;
FORMAT HC040B $TH040C.;
FORMAT HC040B_F $TMA_2F.;
FORMAT HC190 $TH190C.;
FORMAT HC190_F $TMA_3F.;
FORMAT HC200 $TH190C.;
FORMAT HC200_F $TMA_4F.;
FORMAT HC221 $TH221C.;
FORMAT HC221_F $TMA_5F.;
FORMAT HC230 $TH040C.;
FORMAT HC230_F $TMA_6F.;
FORMAT HC240 $TH190C.;
FORMAT HC240_F $TMA_4F.;
FORMAT HC250 $TH250C.;
FORMAT HC250_F $TMA_7F.;
FORMAT HC300 $TH300C.;
FORMAT HC300_F $TMA_3F.;



RUN;
/* FIN PROGRAMA: Microdatos en SAS: ECV_Th_2022.sas*/
