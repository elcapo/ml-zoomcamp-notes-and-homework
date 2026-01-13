/**********************************************************************************************************************					
Instituto Nacional de Estadística (INE) www.ine.es					
***********************************************************************************************************************					
					
DESCRIPCIÓN:					
Este programa genera un fichero SAS con formatos, partiendo de un fichero sin ellos.					
					
Consta de las siguientes partes:					
	* 1. Definir la librería de trabajo --> Libname				
	* 2. Definición de formatos --> PROC FORMAT				
	* 3. Vincular formatos a la base de datos --> PASO data				
					
 Entrada:                                                           					
     - Fichero SAS sin formatos: 	 ECV_Tr_2022.sas7bdat				
 Salida:                                                           					
     - Fichero SAS con formatos: 	 ECV_Tr_2022_conFormato.sas7bdat				
					
Donde:					
	* Operación: Encuesta de Condiciones de Vida
    * Fichero de datos básicos de la persona_Fichero R(ECV_Tr)
	* Periodo: 2022
					
************************************************************************************************************************/					
		
/* Directorio de trabajo para la operación estadística */
*%let siglas_periodo = ECV_Tr_2022;
*%let conFormato = _conFormato;
					
/*1) Definir la librería de trabajo: introducir el directorio que desee como librería					
(se da como ejemplo 'C:\Mis resultados'), y copiar en ese directorio el fichero sas "ECV_Tr_2022.sas7bdat"*/					
					
*libname ROutput 'C:\Mis resultados';	

options fmtsearch = (ROutput ROutput.cat1);

* 2) DEFINICIÓN DE FORMATOS;
PROC FORMAT LIBRARY=ROutput.cat1;
*Tablas1;
value $T_Sexo

"1"="Hombre"
"2"="Mujer"
;
value $T_SiNo

"1"="Sí"
"0"="No"
"9"="No aplicable"
;
value $T_Flag

"-2"="No aplicable (según apartado)"
"-1"="No consta"
"1"="Variable completada"
;
value $TR200B

"1"="Vive actualmente en el hogar"
"2"="Ausente temporalmente"
;
value $TR211B

"1"="Trabajando"
"2"="Parado"
"3"="Jubilado o jubilado anticipadamente"
"4"="Incapacitado permanente para trabajar"
"5"="Estudiante, escolar o en formación"
"6"="Dedicado a las labores del hogar, cuidado de niños u otras personas"
"8"="Otra clase de inactividad económica"
;
value $TR280B

"1"="España"
"2"="Extranjero (resto de la Unión Europea -a partir de ECV2020, UE-27, tras la salida del Reino Unido de la UE )"
"3"="Extranjero (resto del mundo)"
;

*Tablas4;

value $TRB220F

"-2"="El padre no es miembro del hogar o no tiene padre"
"-1"="No consta"
"1"="Variable completada"
;
value $TRB230F

"-2"="La madre no es miembro del hogar o no tiene madre"
"-1"="No consta"
"1"="Variable completada"
;
value $TRB240F

"-2"="No tiene cónyuge o pareja o ésta no es miembro del hogar"
"-1"="No consta"
"1"="Variable completada"
;
value $TRL010F

"-2"="No aplicable (N.A.) (ya que no tiene la edad para ser admitida en estos centros o cursa estudios primarios o tiene más de 12 años)"
"-1"="No consta"
"1"="Variable completada"
;
value $TRL020F

"-2"="N.A. (ya que no tiene la edad para ser admitida en estos centros o tiene más de 12 años)"
"-1"="No consta"
"1"="Variable completada"
;
value $TRL030F

"-2"="N.A. (la persona no está en educación infantil, ni primaria, o tiene más de 12)"
"-1"="No consta"
"1"="Variable completada"
;
value $TRL070F

"-2"="N.A. (la persona tiene más de 12 años en el momento de la entrevista)"
"-1"="No consta"
"1"="Variable completada"
;

*TablasMI;
value $TR010CH

"1"="Muy buena"
"2"="Buena"
"3"="Regular"
"4"="Mala"
"5"="Muy mala"
;
value $TMI_1F

"-4"="No aplicable (persona de 16 o más años)"
"-1"="No consta"
"1"="Variable completada"
;
value $TR020CH

"1"="Es demasiado caro"
"2"="No hay transporte público en la zona"
"3"="La estación o la parada del autobús es de difícil acceso"
"4"="Frecuencia de paso demasiado baja u horario inadecuado"
"5"="Tiempo de viaje demasiado largo"
"6"="Preocupaciones de seguridad"
"7"="Otras razones"
;

*TablasMA;

value $TR370C

"1"="Sí"
"2"="No"
;
value $TMA_1F

"-4"="No aplicable (la persona no utiliza servicios de cuidado de niños o no asiste a preescolar ni a la escuela)"
"-2"="No aplicable (la persona tiene más de 12 años en el momento de la entrevista)"
"-1"="No consta"
"1"="Variable completada"
;
value $TMA_2F

"-4"="No aplicable (la persona no utiliza servicios de cuidado de niños o no va a la guardería ni asiste a preescolar ni a la escuela)"
"-2"="No aplicable (la persona tiene más de 12 años en el momento de la entrevista)"
"-1"="No consta"
"1"="Variable completada"
;
value $TMA_3F

"-2"="No aplicable (la persona tiene más de 12 años en el momento de la entrevista)"
"-1"="No consta"
"1"="Variable completada"
;
value $TR390C

"1"="El hogar no puede permitírselo"
"2"="No hay plazas disponibles"
"3"="Hay plazas disponibles, pero no cerca"
"4"="Hay plazas disponibles, pero no tienen un horario adecuado"
"5"="Hay plazas disponibles, pero la calidad de los servicios no es satisfactoria"
"6"="Otras razones"
;
value $TMA_4F

"-4"="No aplicable (la persona no necesita o no necesita más servicios de cuidado de niños (RC380 = 2))"
"-2"="No aplicable (la persona tiene más de 12 años en el momento de la entrevista)"
"-1"="No consta"
"1"="Variable completada"
;


* 3) VINCULAR FORMATOS A LA BASE DE DATOS;

	DATA ROutput.&siglas_periodo.&conFormato;
		set ROutput.&siglas_periodo;





FORMAT RB050_F $T_Flag.;

FORMAT RB080_F $T_Flag.;

FORMAT RB081_F $T_Flag.;

FORMAT RB082_F $T_Flag.;
FORMAT RB090 $T_Sexo.;
FORMAT RB090_F $T_Flag.;
FORMAT RB200 $TR200B.;
FORMAT RB200_F $T_Flag.;
FORMAT RB211 $TR211B.;
FORMAT RB211_F $T_Flag.;

FORMAT RB220_F $TRB220F.;

FORMAT RB230_F $TRB230F.;

FORMAT RB240_F $TRB240F.;
FORMAT RB280 $TR280B.;
FORMAT RB280_F $T_Flag.;
FORMAT RB290 $TR280B.;
FORMAT RB290_F $T_Flag.;

FORMAT RL010_F $TRL010F.;

FORMAT RL020_F $TRL020F.;

FORMAT RL030_F $TRL030F.;

FORMAT RL040_F $TRL070F.;

FORMAT RL050_F $TRL070F.;

FORMAT RL060_F $TRL070F.;

FORMAT RL070_F $TRL070F.;
FORMAT vrLOWJOB $T_SiNo.;
FORMAT vrEU2020 $T_SiNo.;
FORMAT vrMATSOCDEP $T_SiNo.;
FORMAT vrLOWJOB_nuevo $T_SiNo.;
FORMAT vrEU2030_nuevo $T_SiNo.;
FORMAT RCH010 $TR010CH.;
FORMAT RCH010_F $TMI_1F.;
FORMAT RCH020 $TR020CH.;
FORMAT RCH020_F $TMI_1F.;
FORMAT RC370 $TR370C.;
FORMAT RC370_F $TMA_1F.;
FORMAT RC370B $TR370C.;
FORMAT RC370B_F $TMA_2F.;
FORMAT RC380 $TR370C.;
FORMAT RC380_F $TMA_3F.;
FORMAT RC390 $TR390C.;
FORMAT RC390_F $TMA_4F.;


RUN;
/* FIN PROGRAMA: Microdatos en SAS: ECV_Tr_2022.sas*/
