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
     - Fichero SAS sin formatos: 	 EPA_2021T1.sas7bdat				
 Salida:                                                           					
     - Fichero SAS con formatos: 	 EPA_2021T1_conFormato.sas7bdat				
					
Donde:					
	* Operación: Encuesta de Población Activa
	* Periodo: 2021T1		
					
************************************************************************************************************************/					
		
/* Directorio de trabajo para la operación estadística */
*%let siglas_periodo = EPA_2021T1;
*%let conFormato = _conFormato;
					
/*1) Definir la librería de trabajo: introducir el directorio que desee como librería					
(se da como ejemplo 'C:\Mis resultados'), y copiar en ese directorio el fichero sas "EPA_2021T1.sas7bdat"*/					
					
*libname ROutput 'C:\Mis resultados';	

options fmtsearch = (ROutput ROutput.cat1);

* 2) DEFINICIÓN DE FORMATOS;
PROC FORMAT LIBRARY=ROutput.cat1;

*TABLA1;
value $TNIVEL

"1"="Persona de 16 o más años"
"2"="Menor de 16 años"
;
value $T5EDAD

"00"="0 a 4 años"
"05"="5 a 9 años"
"10"="10 a 15 años"
"16"="16 a 19 años"
"20"="20 a 24 años"
"25"="25 a 29 años"
"30"="30 a 34 años"
"35"="35 a 39 años"
"40"="40 a 44 años"
"45"="45 a 49 años"
"50"="50 a 54 años"
"55"="55 a 59 años"
"60"="60 a 64 años"
"65"="65 o más años"
;
value $TRELPP

"1"="Persona de referencia (p.r.)"
"2"="Cónyuge o pareja de la p.r."
"3"="Hijo/a, hijastro/a (de la p.r o pareja del mismo)"
"4"="Yerno , nuera de la p.r. o de su pareja (o pareja del hijo/a, hijastro/a)"
"5"="Nieto/a, nieto/a de la p.r. o de su pareja (incluye nietastros/as e ambos)"
"6"="Padre, madre, suegro/a de la p.r o pareja de los mismos (padrastro, madrastra)"
"7"="Otro pariente de la p.r (o pareja del mismo)"
"8"="Persona del servicio doméstico"
"9"="Sin parentesco con la p.r."
;
value $TSEXO

"1"="Hombre"
"6"="Mujer"
;
value $TECIV

"1"="Soltero"
"2"="Casado"
"3"="Viudo"
"4"="Separado o divorciado"
;
value $TNACIO

"1"="Española"
"2"="Española y doble nacionalidad"
"3"="Extranjera"
;
value $TNFORMA

"AN"="Analfabetos (código 01 en CNED-2014), (código 80 en CNED-2000)"
"P1"="Educación primaria incompleta (código 02 en CNED-2014), (código 11 en CNED-2000)"
"P2"="Educación primaria (código 10 en CNED-2014), (código 12 en CNED 2000)"
"S1"="Primera etapa de educación secundaria (códigos 21-24 en CNED-2014), (códigos 21-23, 31, 36* en CNED-2000)"
"SG"="Segunda etapa de educación secundaria. Orientación general (código 32 en CNED-2014), (código 32 en CNED-2000)"
"SP"="Segunda etapa de educación secundaria. Orientación profesional (incluye educación postsecundaria no superior) (códigos 33-35, 38**, 41 en CNED-2014), (códigos 33, 34, 41 en CNED-2000)"
"SU"="Educación superior (códigos 51, 52, 61-63, 71-75, 81 en CNED-2014), (códigos 50-56, 59, 61 en CNED-2000)"
;
value $TCURSR

"1"="Sí"
"2"="Estudiante en vacaciones"
"3"="No"
;
value $TNCURSR

"PR"="Educación primaria (códigos 11-13 en CNED-2014), (códigos 11-13 en CNED-2000)"
"S1"="Primera etapa de educación secundaria (códigos 21-23 en CNED-2014), (códigos 22, 23, 36** en CNED-2000)"
"SG"="Segunda etapa de educación secundaria. Orientación general (códigos 31, 32 en CNED-2014), (código 32 en CNED-2000)"
"SP"="Segunda etapa de educación secundaria. Orientación profesional (incluye educación postsecundaria no superior (códigos 33-35*, 36-37, 38***, 41 en CNED-2014), (códigos 33, 34 en CNED-2000)"
"SU"="Educación superior (códigos 51, 52, 61-63, 71-75, 81 en CNED-2014), (códigos 50-52, 54-56, 59, 61 en CNED-2000)"
;
value $TOBJNR

"1"="Proporcionar formación relacionada con la ocupación actual"
"2"="Proporcionar formación relacionada con un posible empleo futuro"
"3"="Proporcionar formación no relacionada con el trabajo (interés personal u otros motivos)"
;
value $TSINO

"1"="Sí"
"6"="No"
;
value $TSINONS

"1"="Sí"
"6"="No"
"0"="No sabe"
;
value $TRZNOTB

"01"="Vacaciones o dias de permiso"
"02"="Permiso por nacimiento de un hijo"
"03"="Excedencia por nacimiento de un hijo"
"04"="Enfermedad, accidente o incapacidad temporal del encuestado"
"05"="Jornada de verano, horario variable, flexible o similar"
"06"="Actividades de representación sindical"
"07"="Nuevo empleo en el que aún no había empezado a trabajar"
"08"="Fijo discontinuo o trabajador estacional en la época de menor actividad"
"09"="Mal tiempo"
"10"="Paro parcial por razones técnicas o económicas"
"11"="Se encuentra en expediente de regulación de empleo"
"12"="Huelga o conflicto laboral"
"13"="Haber recibido enseñanza o formación relacionada con el trabajo"
"14"="Razones personales o responsabilidades familiares"
"15"="Otras razones"
"00"="No sabe"
;
value $TVINCUL

"01"="Vacaciones o día de permiso; Permiso por nacimiento de un hijo; Enfermedad, accidente o incapacidad temporal del encuestado; Jornada de verano, horario variable, flexible o similar; Haber recibido enseñanza o formación relacionada con el trabajo"
"02"="Excedencia por cuidado de hijos con vinculación fuerte con el empleo."
"04"="Fijo discontinuo o trabajador estacional en la época de menor actividad, que realiza regularmente alguna tarea relacionada con el empleo estacional"
"05"="Actividades de representación sindical; Mal tiempo; Expediente de regulación de empleo; Paro parcial por razones técnicas; Encontrarse en expediente de regulación de empleo; Huelga o conflicto laboral; Razones personales o responsabilidades familiares, Otras razones; No sabe. En todos los casos, siempre que mantengan un vínculo fuerte con el empleo."
"07"="Excedencia por cuidado de hijos, con vinculación débil con el empleo"
"08"="Fijo discontinuo o trabajador estacional en la época de menor actividad, que ya no realiza regularmente ninguna tarea relacionada con el empleo estacional"
"09"="Actividades de representación sindical; Mal tiempo; Expediente de regulación de empleo; Paro parcial por razones técnicas; Encontrarse en expediente de regulación de empleo; Huelga o conflicto laboral; Razones personales o responsabilidades familiares, Otras razones; No sabe. En todos los casos, siempre que su vinculación con el empleo sea débil."
"11"="Nuevo empleo en el que aún no había empezado a trabajar"
;
value $TNUEVEM

"1"="Sí, se incorporará en un plazo inferior o igual a tres meses"
"2"="Sí, se incorporará en un plazo superior a tres meses"
"3"="No"
;
value $TOCUP

"0"="Ocupaciones militares (códigos CNO-2011). Fuerzas armadas (códigos CNO-1994)"
"1"="Directores y gerentes (códigos CNO-2011). Dirección de las empresas y de las Administraciones Públicas (códigos CNO-1994)"
"2"="Técnicos y Profesionales científicos e intelectuales (códigos CNO-2011)"
"3"="Técnicos y Profesionales de apoyo (códigos CNO-2011)"
"4"="Empleados contables, administrativos y otros empleados de oficina (códigos CNO-2011). Empleados de tipo administrativo (códigos CNO-1994)"
"5"="Trabajadores de servicios de restauración, personales, protección y vendedores de comercio (códigos CNO-2011)"
"6"="Trabajadores cualificados en el sector agrícola, ganadero, forestal y pesquero (códigos CNO-2011).Trabajadores cualificados en la agricultura y en la pesca (códigos CNO-1994)"
"7"="Artesanos y trabajadores cualificados de las industrias manufactureras y la construcción (excepto operadores de instalaciones y maquinaria (códigos CNO-2011). Artesanos y trabajadores cualificados de las industrias manufactureras, la construcción, y la minería, excepto operadores de instalaciones y maquinaria (códigos CNO-1994)"
"8"="Operadores de instalaciones y maquinaria, y montadores (códigos CNO-2011)"
"9"="Ocupaciones elementales (códigos CNO-2011). Trabajadores no cualificados (códigos CNO-1994)"
;
value $TACTIV

"0"="Agricultura, ganadería, silvicultura y pesca (códigos CNAE-09: 01, 02 y 03), (códigos CNAE-93: 01, 02 y 05)"
"1"="Industria de la alimentación, textil, cuero, madera y papel (códigos CNAE-09: del 10 al 18), (códigos CNAE-93 del 15 al 22)"
"2"="Industrias extractivas, refino de petróleo, industria química, farmaceutica, industria del caucho y materias plásticas, suministro energía eléctrica, gas, vapor y aire acondicionado, suministro de agua, gestión de residuos. Metalurgia (códigos CNAE-09: del 05 al 09, del 19 al 25, 35 y del 36 al 39), (códigos CNAE-93: del 10 al 14, del 23 al 28, 40 y 41)"
"3"="Construcción de maquinaria, equipo eléctrico y material de transporte. Instalación y reparación industrial (códigos CNAE-09 del 26 al 33), (códigos CNAE-93 del 29 al 37)"
"4"="Construcción (códigos CNAE-09: del 41 al 43), (código CNAE-93: 45)"
"5"="Comercio al por mayor y al por menor y sus instalaciones y reparaciones. Reparación de automóviles, hostelería (códigos CNAE-09: del 45 al 47, 55 y 56), (códigos CNAE-93: 50, 51, 52 y 55)"
"6"="Transporte y almacenamiento. Información y comunicaciones (códigos CNAE-09 del 49 al 53 y del 58 al 63), (códigos CNAE-93 del 60 al 64)"
"7"="Intermediación financiera, seguros, actividades inmobiliarias, servicios profesionales, científicos, administrativos y otros (códigos CNAE-09: del 64 al 66, 68, del 69 al 75 y del 77 al 82), (códigos CNAE-93 del 65 al 67 y del 70 al 74)"
"8"="Administración Pública, educación y actividades sanitarias (códigos CNAE-09: 84, 85 y del 86 al 88), (códigos CNAE-93: 75, 80 y 85)"
"9"="Otros servicios (códigos CNAE-09: del 90 al 93, del 94 al 96, 97y 99), (códigos CNAE-93: del 90 al 93, 95 y 99)"
;
value $TSITUAC

"01"="Empresario con asalariados"
"03"="Trabajador independiente o empresario sin asalariados"
"05"="Miembro de una cooperativa"
"06"="Ayuda en la empresa o negocio familiar"
"07"="Asalariado sector público"
"08"="Asalariado sector privado"
"09"="Otra situación"
;
value $TADMTB

"1"="Administración central"
"2"="Administración de la Seguridad Social"
"3"="Administración de Comunidad Autónoma"
"4"="Administración local"
"5"="Empresas públicas e Instituciones financieras públicas"
"6"="Otro tipo"
"0"="No sabe"
;
value $T1DUCON

"1"="Indefinido"
"6"="Temporal"
;
value $T2DUCON

"1"="Permanente"
"6"="Discontinuo"
;
value $T3DUCON

"01"="Eventual por circunstancias de la producción"
"02"="De formación o aprendizaje"
"03"="Estacional o de temporada"
"04"="Cubre un período de prueba"
"05"="Cubre la ausencia total o parcial de otro trabajador"
"06"="Para obra o servicio determinado"
"07"="Verbal no incluido en las opciones anteriores"
"08"="Otro tipo"
"09"="De prácticas (becarios, períodos de prácticas, asistentes de investigación, etc.)"
"00"="No sabe"
;
value $T1PARCO

"1"="Completa"
"6"="Parcial"
;
value $T2PARCO

"01"="Seguir cursos de enseñanza o formación"
"02"="Enfermedad o incapacidad propia"
"03"="Responsabilidades de cuidado de hijos u otros familiares"
"04"="Otras razones familiares o personales"
"05"="No haber podido encontrar un trabajo de jornada completa"
"06"="No querer un trabajo de jornada completa"
"07"="Otras razones"
"00"="Desconoce el motivo"
;
value $TMASHOR

"1"="Sí"
"2"="No, desearía trabajar menos horas con reducción proporcional de salario"
"3"="No"
;
value $TRZNDIS

"1"="Tener que completar estudios o formación"
"2"="Responsabilidades de cuidado de hijos u otros familiares"
"3"="Por enfermedad o incapacidad propia"
"4"="Por otras razones"
;
value $TRZNDSH

"01"="Tener que completar estudios o formación"
"02"="Responsabilidades de cuidado de hijos u otros familiares"
"03"="Enfermedad o incapacidad propia"
"04"="Otras razones"
"05"="Por no poder dejar su empleo actual debido al periodo de preaviso"
;
value $TFOBACT

"1"="Métodos activos de búsqueda de empleo"
"6"="Métodos no activos de búsqueda de empleo"
;
value $TNBUSCA

"01"="No hay empleo adecuado disponible"
"02"="Está afectado por una regulación de empleo"
"03"="Por enfermedad o incapacidad propia"
"04"="Responsabilidades de cuidado de hijos u otros familiares"
"05"="Tiene otras responsabilidades familiares o personales"
"06"="Está cursando estudios o recibiendo formación"
"07"="Está jubilado"
"08"="Otras razones"
"00"="No sabe"
;
value $TRZULT

"01"="Despido o supresión del puesto (incluye regulación de empleo)"
"02"="Fin del contrato (incluye los fijos-discontinuos y los trabajos estacionales) s"
"03"="Enfermedad o incapacidad propia"
"04"="Realizar estudios o recibir formación"
"05"="Responsabilidades de cuidado de hijos u otros familiares"
"06"="Otras razones familiares o personales"
"07"="Jubilación anticipada"
"08"="Jubilación normal"
"09"="Otras razones (incluye el cese en una actividad propia y por voluntad propia)"
"00"="No sabe"
;
value $TITBU

"01"="Menos de 1 mes"
"02"="De 1 a < 3 meses"
"03"="De 3 a < 6 meses"
"04"="De 6 meses a < 1 año"
"05"="De 1 año a < 1 año y medio"
"06"="De 1 año y medio a < 2 años"
"07"="De 2 a < 4 años"
"08"="4 años o más"
;
value $TOFEMP

"1"="Estaba inscrito como demandante y recibía algún tipo de prestación"
"2"="Estaba inscrito como demandante sin recibir subsidio o prestación por desempleo"
"3"="No estaba inscrito como demandante"
"4"="No contesta / No sabe"
;
value $TSID

"01"="Estudiante (aunque esté de vacaciones)"
"02"="Percibía una pensión de jubilación o unos ingresos de prejubilación"
"03"="Dedicado a las labores del hogar"
"04"="Incapacitado permanente"
"05"="Percibiendo una pensión distinta a la de jubilación (o prejubilación)"
"06"="Realizando sin remuneración trabajos sociales, actividades benéficas…"
"07"="Otras situaciones"
"00"="No sabe / No refiere estado de inactividad"
;
value $TSIDAC

"1"="Trabajando"
"2"="Buscando empleo"
;
value $TAOI

"03"="Ocupados subempleados por insuficiencia de horas"
"04"="Resto de ocupados"
"05"="Parados que buscan primer empleo"
"06"="Parados que han trabajado antes"
"07"="Inactivos 1 (desanimados)"
"08"="Inactivos 2 (junto con los desanimados forman los activos potenciales)"
"09"="Inactivos 3 (resto de inactivos)"
;

*TABLA2;
value $TCCAA

"01"="Andalucía"
"02"="Aragón"
"03"="Asturias, Principado de"
"04"="Balears, Illes"
"05"="Canarias"
"06"="Cantabria"
"07"="Castilla y León"
"08"="Castilla-La Mancha"
"09"="Cataluña"
"10"="Comunitat Valenciana"
"11"="Extremadura"
"12"="Galicia"
"13"="Madrid, Comunidad de"
"14"="Murcia, Región de"
"15"="Navarra, Comunidad Foral de"
"16"="País Vasco"
"17"="Rioja, La"
"51"="Ceuta"
"52"="Melilla"
;
value $TPROV

"01"="Araba/Álava"
"02"="Albacete"
"03"="Alicante/Alacant"
"04"="Almería"
"05"="Ávila"
"06"="Badajoz"
"07"="Balears, Illes"
"08"="Barcelona"
"09"="Burgos"
"10"="Cáceres"
"11"="Cádiz"
"12"="Castellón /Castelló"
"13"="Ciudad Real"
"14"="Córdoba"
"15"="Coruña, A"
"16"="Cuenca"
"17"="Girona"
"18"="Granada"
"19"="Guadalajara"
"20"="Gipuzkoa"
"21"="Huelva"
"22"="Huesca"
"23"="Jaén"
"24"="León"
"25"="Lleida"
"26"="Rioja, La"
"27"="Lugo"
"28"="Madrid"
"29"="Málaga"
"30"="Murcia"
"31"="Navarra"
"32"="Ourense"
"33"="Asturias"
"34"="Palencia"
"35"="Palmas, Las"
"36"="Pontevedra"
"37"="Salamanca"
"38"="Santa Cruz de Tenerife"
"39"="Cantabria"
"40"="Segovia"
"41"="Sevilla"
"42"="Soria"
"43"="Tarragona"
"44"="Teruel"
"45"="Toledo"
"46"="Valencia/València"
"47"="Valladolid"
"48"="Bizkaia"
"49"="Zamora"
"50"="Zaragoza"
"51"="Ceuta"
"52"="Melilla"
;
*TABLA3;
value $TREGNAP

"115"="UE- 15"
"125"="UE- 25 (no UE-15)"
"128"="UE- 28 (no UE-27)"
"100"="Resto de Europa"
"200"="África"
"300"="América del Norte"
"310"="Centroamérica y Caribe"
"350"="Sudamérica"
"400"="Asia Oriental (Lejano Oriente)"
"410"="Asia Occidental (Oriente Próximo)"
"420"="Asia del Sur y Sudoeste"
"500"="Oceanía"
"999"="Apátridas"
;
value $TREGNA

"115"="UE- 15"
"125"="UE- 25 (no UE-15)"
"128"="UE- 28 (no UE-27)"
"100"="Resto de Europa"
"200"="África"
"300"="América del Norte"
"310"="Centroamérica y Caribe"
"350"="Sudamérica"
"400"="Asia Oriental (Lejano Oriente)"
"410"="Asia Occidental (Oriente Próximo)"
"420"="Asia del Sur y Sudoeste"
"500"="Oceanía"
;
value $TREGEST

"115"="UE- 15 (Excepto Francia y Portugal)"
"125"="UE- 25 (no UE-15)"
"128"="UE- 28 (no UE-27)"
"100"="Resto de Europa (Excepto Andorra)"
"200"="África (Excepto Marruecos)"
"300"="América del Norte"
"310"="Centroamérica y Caribe"
"350"="Sudamérica"
"400"="Asia Oriental (Lejano Oriente)"
"410"="Asia Occidental (Oriente Próximo)"
"420"="Asia del Sur y Sudoeste"
"500"="Oceanía"
"600"="Portugal"
"610"="Francia"
"620"="Andorra"
"630"="Marruecos"
;

*TABLAS4;
value N_EDEST

00="No sabe la fecha en la que alcanzó el máximo nivel de estudios"
;
value $T_ORDEN

"00"="No tiene o no reside en la vivienda"
;
value N_ANORE

00="Menos de un año en España"
;
value $T_2HORA

"99"="No puede precisar /no sabe"
;
value $T_CONTM

"96"="96 meses o más"
"00"="Desconoce la respuesta pero es al menos un mes"
;
value $T_CONTD

"99"="No sabe"
"00"="Desconoce la respuesta pero es menos de un mes"
;
value $THORAS

"9999"="No sabe (incluye "No tiene horas pactadas por contrato")"
;
value $T2HORAS

"9999"="No sabe horas:minutos"
"9900"="Las horas varían de una semana a otra/no puede dar una estimación"
;
value $THORASE

"9999"="No sabe horas:minutos"
"9859"="Más de 98:00 horas en la semana de referencia (código específico pandemia COVID19)"
"0000"="No trabajó durante la semana de referencia"
;
value $TEXTRAO

"9999"="No puede precisar /No sabe"
"0000"="No hizo horas extra durante la semana de referencia"
;
value N_2DIAS

99="No sabe días de ausencia"
;
*TABLA5;
value $T_CICLO

"194"="2021T1"
"195"="2021T2"
"196"="2021T3"
"197"="2021T4"
"198"="2022T1"
"199"="2022T2"
"200"="2022T3"
"201"="2022T4"
"202"="2023T1"
"203"="2023T2"
"204"="2023T3"
"205"="2023T4"
"206"="2024T1"
"207"="2024T2"
"208"="2024T3"
"209"="2024T4"
"210"="2025T1"
"211"="2025T2"
"212"="2025T3"
"213"="2025T4"
"214"="2026T1"
"215"="2026T2"
"216"="2026T3"
"217"="2026T4"
"218"="2027T1"
"219"="2027T2"
"220"="2027T3"
"221"="2027T4"
"222"="2028T1"
"223"="2028T2"
"224"="2028T3"
"225"="2028T4"
"226"="2029T1"
"227"="2029T2"
"228"="2029T3"
"229"="2029T4"
"230"="2030T1"
"231"="2030T2"
"232"="2030T3"
"233"="2030T4"
"234"="2031T1"
"235"="2031T2"
"236"="2031T3"
"237"="2031T4"
"238"="2032T1"
"239"="2032T2"
"240"="2032T3"
"241"="2032T4"
"242"="2033T1"
"243"="2033T2"
"244"="2033T3"
"245"="2033T4"
"246"="2034T1"
"247"="2034T2"
"248"="2034T3"
"249"="2034T4"
"250"="2035T1"
"251"="2035T2"
"252"="2035T3"
"253"="2035T4"
"254"="2036T1"
"255"="2036T2"
"256"="2036T3"
"257"="2036T4"
"258"="2037T1"
"259"="2037T2"
"260"="2037T3"
"261"="2037T4"
"262"="2038T1"
"263"="2038T2"
"264"="2038T3"
"265"="2038T4"
"266"="2039T1"
"267"="2039T2"
"268"="2039T3"
"269"="2039T4"
"270"="2040T1"
"271"="2040T2"
"272"="2040T3"
"273"="2040T4"
"274"="2041T1"
"275"="2041T2"
"276"="2041T3"
"277"="2041T4"
"278"="2042T1"
"279"="2042T2"
"280"="2042T3"
"281"="2042T4"
"282"="2043T1"
"283"="2043T2"
"284"="2043T3"
"285"="2043T4"
"286"="2044T1"
"287"="2044T2"
"288"="2044T3"
"289"="2044T4"
"290"="2045T1"
"291"="2045T2"
"292"="2045T3"
"293"="2045T4"
"294"="2046T1"
"295"="2046T2"
"296"="2046T3"
"297"="2046T4"
"298"="2047T1"
"299"="2047T2"
"300"="2047T3"
"301"="2047T4"
"302"="2048T1"
"303"="2048T2"
"304"="2048T3"
"305"="2048T4"
"306"="2049T1"
"307"="2049T2"
"308"="2049T3"
"309"="2049T4"
"310"="2050T1"
"311"="2050T2"
"312"="2050T3"
"313"="2050T4"
;






* 3) VINCULAR FORMATOS A LA BASE DE DATOS;

	DATA ROutput.&siglas_periodo.&conFormato;
		set ROutput.&siglas_periodo;

FORMAT CICLO $T_CICLO.;
FORMAT CCAA $TCCAA.;
FORMAT PROV $TPROV.;

FORMAT NIVEL $TNIVEL.;

FORMAT EDAD1 $T5EDAD.;
FORMAT RELPP1 $TRELPP.;
FORMAT SEXO1 $TSEXO.;
FORMAT NCONY $T_ORDEN.;
FORMAT NPADRE $T_ORDEN.;
FORMAT NMADRE $T_ORDEN.;

FORMAT ECIV1 $TECIV.;
FORMAT PRONA1 $TPROV.;
FORMAT REGNA1 $TREGNA.;
FORMAT NAC1 $TNACIO.;
FORMAT EXREGNA1 $TREGNAP.;
FORMAT ANORE1 N_ANORE.;
FORMAT NFORMA $TNFORMA.;

FORMAT EDADEST N_EDEST.;
FORMAT CURSR $TCURSR.;
FORMAT NCURSR $TNCURSR.;
FORMAT CURSNR $TCURSR.;
FORMAT OBJFORM $TOBJNR.;

FORMAT TRAREM $TSINO.;
FORMAT AYUDFA $TSINO.;
FORMAT AUSENT $TSINO.;
FORMAT RZNOTB $TRZNOTB.;
FORMAT VINCUL $TVINCUL.;
FORMAT NUEVEM $TNUEVEM.;
FORMAT OCUP1 $TOCUP.;
FORMAT ACT1 $TACTIV.;
FORMAT SITU $TSITUAC.;
FORMAT SP $TADMTB.;
FORMAT DUCON1 $T1DUCON.;
FORMAT DUCON2 $T2DUCON.;
FORMAT DUCON3 $T3DUCON.;
FORMAT TCONTM $T_CONTM.;
FORMAT TCONTD $T_CONTD.;


FORMAT PROEST $TPROV.;
FORMAT REGEST $TREGEST.;
FORMAT PARCO1 $T1PARCO.;
FORMAT PARCO2 $T2PARCO.;
FORMAT HORASP $THORAS.;
FORMAT HORASH $T2HORAS.;
FORMAT HORASE $THORASE.;
FORMAT EXTRA $TSINO.;
FORMAT EXTPAG $TEXTRAO.;
FORMAT EXTNPG $TEXTRAO.;

FORMAT TRAPLU $TSINO.;
FORMAT OCUPLU1 $TOCUP.;
FORMAT ACTPLU1 $TACTIV.;
FORMAT SITPLU $TSITUAC.;
FORMAT HOREPLU $THORASE.;
FORMAT MASHOR $TMASHOR.;
FORMAT DISMAS $TSINO.;
FORMAT RZNDISH $TRZNDSH.;
FORMAT HORDES $T_2HORA.;
FORMAT BUSOTR $TSINO.;
FORMAT BUSCA $TSINO.;
FORMAT DESEA $TSINO.;
FORMAT FOBACT $TFOBACT.;
FORMAT NBUSCA $TNBUSCA.;
FORMAT RZULT $TRZULT.;
FORMAT ITBU $TITBU.;
FORMAT DISP $TSINO.;
FORMAT RZNDIS $TRZNDIS.;
FORMAT EMPANT $TSINO.;

FORMAT OCUPA $TOCUP.;
FORMAT ACTA $TACTIV.;
FORMAT SITUA $TSITUAC.;
FORMAT OFEMP $TOFEMP.;
FORMAT SIDI1 $TSID.;
FORMAT SIDI2 $TSID.;
FORMAT SIDI3 $TSID.;
FORMAT SIDAC1 $TSIDAC.;
FORMAT SIDAC2 $TSIDAC.;
FORMAT DAUSVAC N_2DIAS.;
FORMAT DAUSENF N_2DIAS.;
FORMAT DAUSOTR N_2DIAS.;
FORMAT TRAANT $TSINONS.;
FORMAT AOI $TAOI.;


RUN;
/* FIN PROGRAMA: Microdatos en SAS: siglas_periodo.sas*/
