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
     - Fichero SAS sin formatos: 	 ECV_Td_2019.sas7bdat				
 Salida:                                                           					
     - Fichero SAS con formatos: 	 ECV_Td_2019_conFormato.sas7bdat				
					
Donde:					
	* Operaciï¿½n: Encuesta de Condiciones de Vida
    * Fichero Transversal_Datos bï¿½sicos del hogar_Fichero D(ECV_Td)
	* Periodo: 2019
					
************************************************************************************************************************/					
		
/* Directorio de trabajo para la operación estadística */
*%let siglas_periodo = ECV_Td_2022;
*%let conFormato = _conFormato;
					
/*1) Definir la librería de trabajo: introducir el directorio que desee 					
(se da como ejemplo 'C:\Mis resultados'), y copiar en ese directorio el fichero sas "ECV_Td_2019.sas7bdat"*/					
					
*libname ROutput 'C:\Mis resultados';	

options fmtsearch = (ROutput ROutput.cat1);

* 2) DEFINICION DE FORMATOS;
PROC FORMAT LIBRARY=ROutput.cat1;

value $TD040B

"ES11"="Galicia"
"ES12"="Principado de Asturias"
"ES13"="Cantabria"
"ES21"="País Vasco"
"ES22"="Comunidad Foral de Navarra"
"ES23"="La Rioja"
"ES24"="Aragón"
"ES30"="Comunidad de Madrid"
"ES41"="Castilla y León"
"ES42"="Castilla-La Mancha"
"ES43"="Extremadura"
"ES51"="Cataluña"
"ES52"="Comunitat Valenciana"
"ES53"="Illes Balears"
"ES61"="Andalucía"
"ES62"="Región de Murcia"
"ES63"="Ciudad Autónoma de Ceuta"
"ES64"="Ciudad Autónoma de Melilla"
"ES70"="Canarias"
"ESZZ"="Extra-Regio"

;
value $TD100B

"1"="Zona muy poblada"
"2"="Zona media"
"3"="Zona poco poblada"
;
value $T_Flag

"-2"="No disponible"
"-1"="No consta"
"1"="Variable completada"
;


* 3) VINCULAR FORMATOS A LA BASE DE DATOS;

	DATA ROutput.&siglas_periodo.&conFormato;
		set ROutput.&siglas_periodo;

FORMAT DB040 $TD040B.;
FORMAT DB040_F $T_Flag.;
FORMAT DB090_F $T_Flag.;
FORMAT DB100 $TD100B.;
FORMAT DB100_F $T_Flag.;


RUN;
/* FIN PROGRAMA: Microdatos en SAS: ECV_Td_2019.sas*/
