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
     - Fichero SAS sin formatos: 	 ECV_Tp_2023.sas7bdat				
 Salida:                                                           					
     - Fichero SAS con formatos: 	 ECV_Tp_2023_conFormato.sas7bdat				
					
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
value $T_Sexo

"1"="Hombre"
"2"="Mujer"
;
value $TP190B

"1"="Soltero"
"2"="Casado"
"3"="Separado"
"4"="Viudo"
"5"="Divorciado"
;
value $TP200B

"1"="Sí, con base jurídica"
"2"="Sí, sin base jurídica"
"3"="No"
;
value $TP205B

"1"="Sí, viviendo con cónyuge o pareja"
"2"="No, no viviendo ni con cónyuge ni con pareja"
;
value $TP230B

"1"="España"
"2"="Extranjero (resto de la Unión Europea -a partir de ECV2020, UE-27, tras la salida del Reino Unido de la UE )"
"3"="Extranjero (resto del mundo)"
;
value $TP021E

"00"="Menos que primaria"
"10"="Educación primaria"
"20"="Primera etapa de Educación Secundaria"
"30"="Segunda etapa de Educación Secundaria"
"34"="Para personas de 16 a 34 años: Orientación general"
"35"="Para personas de 16 a 34 años: Orientación profesional"
"40"="Educación postsecundaria no superior"
"45"="Para personas de 16 a 34 años: Orientación profesional"
"50"="Educación superior"
;
value $TP041E

"0"="Menos que primaria"
"100"="Educación primaria"
"200"="Primera etapa de Educación Secundaria"
"340"="Segunda etapa de Educación Secundaria: Orientación general"
"344"="Para personas de 16 a 34 años: Orientación general"
"350"="Segunda etapa de Educación Secundaria: Orientación profesional"
"353"="Para personas de 16 a 34 años: Orientación profesional (sin acceso directo a educación superior)"
"354"="Para personas de 16 a 34 años: Orientación profesional (con acceso directo a educación superior)."
"450"="Educación postsecundaria no superior"
"500"="Educación superior"
;
value $TP032L

"1"="Trabajando (asalariado o trabajador por cuenta propia)"
"2"="Parado"
"3"="Jubilado, retirado, jubilado anticipado o ha cerrado un negocio"
"4"="Incapacitado permanente para trabajar"
"5"="Estudiante, escolar o en formación"
"6"="Dedicado a las labores del hogar, al cuidado de niños u otras personas"
"7"="No consta al no disponer de cuestionario individual"
"8"="Otra clase de inactividad económica"
;
value $TP040L

"1"="Empleador"
"2"="Empresario sin asalariados o trabajador independiente"
"3"="Asalariado"
"4"="Ayuda familiar"
;
value $TPL111A

"a"="Agricultura, ganadería, silvicultura y pesca"
"b"="Industrias extractivas"
"c"="Industria manufacturera"
"d"="Suministro de energía eléctrica, gas, etc."
"e"="Suministro de agua. Gestión residuos"
"f"="Construcción"
"g"="Comercio, reparación de vehículos de motor"
"h"="Transporte y almacenamiento"
"i"="Hostelería"
"j"="Información y comunicaciones"
"k"="Actividades financieras y de seguros"
"l"="Actividades inmobiliarias"
"m"="Actividades profesionales, científicas y técnicas"
"n"="Actividades administrativas y servicios auxiliares"
"o"="Administración pública y defensa. Seguridad Social"
"p"="Educación"
"q"="Actividades sanitarias y de servicios sociales"
"r"="Actividades artísticas, recreativas"
"s"="Otros servicios"
"t"="Hogares como empleadores de personal doméstico"
"u"="Organismos extraterritoriales, no consta"
;
value $TP141L

"11"="Contrato escrito temporal de duración determinada"
"12"="Contrato verbal temporal de duración determinada"
"21"="Contrato escrito fijo de duración indefinida"
"22"="Contrato verbal fijo de duración indefinida"
;
value $TPSTCN

"1"="Asalariado (tiempo completo)"
"2"="Asalariado (tiempo parcial)"
"3"="Trabajador por cuenta propia (tiempo completo)"
"4"="Trabajador por cuenta propia (tiempo parcial)"
"5"="Parado"
"6"="Estudiante, escolar o en formación"
"7"="Jubilado o retirado"
"8"="Incapacitado permanente para trabajar"
"10"="Dedicado a las labores del hogar, cuidado de niños, etc"
"11"="Otro tipo de inactividad económica"
;
value $TP010H

"1"="Muy bueno"
"2"="Bueno"
"3"="Regular"
"4"="Malo"
"5"="Muy malo"
;
value $TP030H

"1"="Gravemente limitado"
"2"="Limitado pero no gravemente"
"3"="Nada limitado"
;
value $TP050H

"1"="No se lo podía permitir"
"2"="Estaba en una lista de espera o no tenía volante"
"3"="No disponía de tiempo debido al trabajo, al cuidado de niños o de otras personas"
"4"="Demasiado lejos para viajar/sin medios de transporte"
"5"="Miedo a los médicos, a los hospitales"
"6"="Quiso esperar y ver si el problema mejoraba por sí solo"
"7"="No conocía a ningún médico o especialista"
"8"="Otras razones"
;
value $TP070H

"1"="No se lo podía permitir"
"2"="Estaba en una lista de espera o no tenía volante"
"3"="No disponía de tiempo debido al trabajo, al cuidado de niños o de otras personas"
"4"="Demasiado lejos para viajar/sin medios de transporte"
"5"="Miedo a los médicos, a los hospitales"
"6"="Quiso esperar y ver si el problema mejoraba por sí solo"
"7"="No conocía a ningún médico o especialista"
"8"="Otras razones"
;
value $TP145L

"1"="Trabajo a tiempo completo"
"2"="Trabajo a tiempo parcial"
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

*Tablas3.......................;
value TP050N

low-0="Beneficios"
0="Sin renta"
0-high="Pérdidas"
;
value $TPIngrs

"99"="Indicadores renta: TPIngrsDetalle"
;

*TablasMA.......................;
value $TP280C

"1"="A diario"
"2"="Todas las semanas, pero no todos los días"
"3"="Todos los meses, pero no todas las semanas"
"4"="Menos de una vez al mes"
"5"="Nunca"
;
value $TMA_1F

"-1"="No consta"
"1"="Variable completada"
;
value $TP290C

"1"="Es demasiado caro"
"2"="No hay transporte público en la zona"
"3"="La estación o la parada del autobús es de difícil acceso"
"4"="Frecuencia de paso demasiado baja u horario inadecuado"
"5"="Tiempo de viaje demasiado largo"
"6"="Preocupaciones de seguridad"
"7"="Otras razones"
;
value $TMA_2F

"-2"="No aplicable (utilizó el transporte público al menos todos los meses (PC280 = 1, 2, 3))"
"-1"="No consta"
"1"="Variable completada"
;
value $TP310C

"1"="Sí"
"2"="No"
"3"="No sabe"
;
value $TMA_3F

"-2"="No aplicable (no está trabajando, ni como asalariado ni como autónomo)"
"-1"="No consta"
"1"="Variable completada"
;
value $TP330C

"1"="No he estado en contacto con ninguna oficina administrativa o servicio público"
"2"="Edad"
"3"="Sexo o identidad de género"
"4"="Discapacidad o problema de salud de larga duración"
"5"="Origen étnico o país de origen"
"6"="Religión"
"7"="Orientación sexual"
"8"="Otras razones (nivel de ingresos, profesión, nivel de estudios, aspecto físico, etc.)"
"9"="No me he sentido discriminado"
;
value $TP340C

"1"="No he intentado alquilar o comprar una casa en los últimos 5 años"
"2"="Edad"
"3"="Sexo o identidad de género"
"4"="Discapacidad o problema de salud de larga duración"
"5"="Origen étnico o país de origen"
"6"="Religión"
"7"="Orientación sexual"
"8"="Otras razones (nivel de ingresos, profesión, nivel de estudios, aspecto físico, etc.)"
"9"="No me he sentido discriminado"
;
value $TP350C

"1"="No he estado en contacto con nadie de un centro educativo"
"2"="Edad"
"3"="Sexo o identidad de género"
"4"="Discapacidad o problema de salud de larga duración"
"5"="Origen étnico o país de origen"
"6"="Religión"
"7"="Orientación sexual"
"8"="Otras razones (nivel de ingresos, profesión, nivel de estudios, aspecto físico, etc.)"
"9"="No me he sentido discriminado"
;
value $TP360C

"1"="Edad"
"2"="Sexo o identidad de género"
"3"="Discapacidad o problema de salud de larga duración"
"4"="Origen étnico o país de origen"
"5"="Religión"
"6"="Orientación sexual"
"7"="Otras razones (nivel de ingresos, profesión, nivel de estudios, aspecto físico, etc.)"
"8"="No me he sentido discriminado"
;




* 3) VINCULAR FORMATOS A LA BASE DE DATOS;

	DATA ROutput.&siglas_periodo.&conFormato;
		set ROutput.&siglas_periodo;





FORMAT PB040_F $T_Flag.;

FORMAT PB100_F $T_Flag.;

FORMAT PB110_F $T_Flag.;

FORMAT PB120_F $T_Flag.;

FORMAT PB140_F $T_Flag.;
FORMAT PB150 $T_Sexo.;
FORMAT PB150_F $T_Flag.;

FORMAT PB160_F $T_Flag.;

FORMAT PB170_F $T_Flag.;

FORMAT PB180_F $T_Flag.;
FORMAT PB190 $TP190B.;
FORMAT PB190_F $T_Flag.;
FORMAT PB200 $TP200B.;
FORMAT PB200_F $T_Flag.;
FORMAT PB205 $TP205B.;
FORMAT PB205_F $T_Flag.;
FORMAT PB230 $TP230B.;
FORMAT PB230_F $T_Flag.;
FORMAT PB240 $TP230B.;
FORMAT PB240_F $T_Flag.;
FORMAT PE010 $T_SiNo.;
FORMAT PE010_F $T_Flag.;
FORMAT PE021 $TP021E.;
FORMAT PE021_F $T_Flag.;
FORMAT PE041 $TP041E.;
FORMAT PE041_F $T_Flag.;
FORMAT PL032 $TP032L.;
FORMAT PL032_F $T_Flag.;
FORMAT PL016 $TP016L.;
FORMAT PL016_F $T_Flag.;
FORMAT PL040A $TP040L.;
FORMAT PL040A_F $T_Flag.;
FORMAT PL040B $TP040L.;
FORMAT PL040B_F $T_Flag.;

FORMAT PL051A_F $T_Flag.;

FORMAT PL051B_F $T_Flag.;

FORMAT PL060_F $T_Flag.;

FORMAT PL073_F $T_Flag.;

FORMAT PL074_F $T_Flag.;

FORMAT PL075_F $T_Flag.;

FORMAT PL076_F $T_Flag.;

FORMAT PL080_F $T_Flag.;

FORMAT PL085_F $T_Flag.;

FORMAT PL086_F $T_Flag.;

FORMAT PL087_F $T_Flag.;

FORMAT PL089_F $T_Flag.;

FORMAT PL090_F $T_Flag.;

FORMAT PL100_F $T_Flag.;
FORMAT PL111AA $TPL111A.;
FORMAT PL111AA_F $T_Flag.;
FORMAT PL111BA $TPL111A.;
FORMAT PL111BA_F $T_Flag.;
FORMAT PL141 $TP141L.;
FORMAT PL141_F $T_Flag.;
FORMAT PL145 $TP145L.;
FORMAT PL145_F $T_Flag.;
FORMAT PL150 $T_SiNo.;
FORMAT PL150_F $T_Flag.;

FORMAT PL200_F $T_Flag.;
FORMAT PL211A $TPSTCN.;
FORMAT PL211A_F $T_Flag.;
FORMAT PL211B $TPSTCN.;
FORMAT PL211B_F $T_Flag.;
FORMAT PL211C $TPSTCN.;
FORMAT PL211C_F $T_Flag.;
FORMAT PL211D $TPSTCN.;
FORMAT PL211D_F $T_Flag.;
FORMAT PL211E $TPSTCN.;
FORMAT PL211E_F $T_Flag.;
FORMAT PL211F $TPSTCN.;
FORMAT PL211F_F $T_Flag.;
FORMAT PL211G $TPSTCN.;
FORMAT PL211G_F $T_Flag.;
FORMAT PL211H $TPSTCN.;
FORMAT PL211H_F $T_Flag.;
FORMAT PL211I $TPSTCN.;
FORMAT PL211I_F $T_Flag.;
FORMAT PL211J $TPSTCN.;
FORMAT PL211J_F $T_Flag.;
FORMAT PL211K $TPSTCN.;
FORMAT PL211K_F $T_Flag.;
FORMAT PL211L $TPSTCN.;
FORMAT PL211L_F $T_Flag.;

FORMAT PL271_F $T_Flag.;
FORMAT PH010 $TP010H.;
FORMAT PH010_F $T_Flag.;
FORMAT PH020 $T_SiNo.;
FORMAT PH020_F $T_Flag.;
FORMAT PH030 $TP030H.;
FORMAT PH030_F $T_Flag.;
FORMAT PH040 $TP040H.;
FORMAT PH040_F $T_Flag.;
FORMAT PH050 $TP050H.;
FORMAT PH050_F $T_Flag.;
FORMAT PH060 $T_SiNo.;
FORMAT PH060_F $T_Flag.;
FORMAT PH070 $TP070H.;
FORMAT PH070_F $T_Flag.;

FORMAT PY010N_F $TPIngrs.;

FORMAT PY020N_F $TPIngrs.;

FORMAT PY021N_F $TPIngrs.;

FORMAT PY035N_F $TPIngrs.;
FORMAT PY050N TP050N.;
FORMAT PY050N_F $TPIngrs.;

FORMAT PY080N_F $TPIngrs.;

FORMAT PY090N_F $TPIngrs.;

FORMAT PY100N_F $TPIngrs.;

FORMAT PY110N_F $TPIngrs.;

FORMAT PY120N_F $TPIngrs.;

FORMAT PY130N_F $TPIngrs.;

FORMAT PY140N_F $TPIngrs.;

FORMAT PY010G_F $TPIngrs.;

FORMAT PY020G_F $TPIngrs.;

FORMAT PY021G_F $TPIngrs.;

FORMAT PY030G_F $TPIngrs.;

FORMAT PY035G_F $TPIngrs.;
FORMAT PY050G TP050N.;
FORMAT PY050G_F $TPIngrs.;

FORMAT PY080G_F $TPIngrs.;

FORMAT PY090G_F $TPIngrs.;

FORMAT PY100G_F $TPIngrs.;

FORMAT PY110G_F $TPIngrs.;

FORMAT PY120G_F $TPIngrs.;

FORMAT PY130G_F $TPIngrs.;

FORMAT PY140G_F $TPIngrs.;
FORMAT PD020 $T_Si2No.;
FORMAT PD020_F $T_Flag.;
FORMAT PD030 $T_Si2No.;
FORMAT PD030_F $T_Flag.;
FORMAT PD050 $T_Si2No.;
FORMAT PD050_F $T_Flag.;
FORMAT PD060 $T_Si2No.;
FORMAT PD060_F $T_Flag.;
FORMAT PD070 $T_Si2No.;
FORMAT PD070_F $T_Flag.;
FORMAT PD080 $T_Si2No.;
FORMAT PD080_F $T_Flag.;

FORMAT PW010_F $T_Flag.;

FORMAT PW191_F $T_Flag.;
FORMAT PC280 $TP280C.;
FORMAT PC280_F $TMA_1F.;
FORMAT PC290 $TP290C.;
FORMAT PC290_F $TMA_2F.;
FORMAT PC310 $TP310C.;
FORMAT PC310_F $TMA_3F.;
FORMAT PC320 $TP310C.;
FORMAT PC320_F $TMA_3F.;
FORMAT PC330 $TP330C.;
FORMAT PC330_F $TMA_1F.;
FORMAT PC340 $TP340C.;
FORMAT PC340_F $TMA_1F.;
FORMAT PC350 $TP350C.;
FORMAT PC350_F $TMA_1F.;
FORMAT PC360 $TP360C.;
FORMAT PC360_F $TMA_1F.;



RUN;
/* FIN PROGRAMA: Microdatos en SAS: ECV_Td_2022.sas*/
