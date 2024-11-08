# Introduccion

Un sistema es una combinacion de elementos que interactuan organizados para lograr uno o mas
propositos.

Configuracion de un sistema son las cacateristicas funcionales y de hard o soft tal como se
establece en una documentacion tecnica o en el mismo producto.

La gestion de configuracion es la disciplica de identificar la configuracion de un sistema en
distintos momentos para documentar y manterner una ttrzabilidad durante su periodo de vida.

El SCM (gestion de conf del soft) es un proceso natural del ciclo de vida del soft.
Es parte del proceso de garantia de calidad del software (SQA).

Las actividades SCM:
- son la gestion y planificacion del SCM
- la identificacion de la configuracion del softw
- el control de la conf de soft
- contabilidad del estado de la conf
- la auditoria del soft
- gestion y entrega de la version de soft

## Desgloce de temas para SCM

### _Gestion del proceso SCM_:

  controla la evolucion e integridad del producto identificando registrando y verificando los
  cambios, de lo que se debe tener en cuenta es:
  + contexto organizacional del SCM:
    para u nproyecto es necesario comprender la ogranizacion de la empresa y sus relaciones.
    Despues te cuenta su vida de que tiene que organizar el software con el hardware para que
    sea coherente con la meta.

  + REstricciones y orientacion:
    el proyecto tiene restricciones y limitaciones que afectan a crear el plan SCM, dados por
    el contrato, las regulaciones y restricciones a nivel corporativo como presupuesto.

El desgloce de la gestion scm

#### Planificacion del SCM:

  debe ser coherente con las restricciones, el contexto, la direccion y la naturaleza del
  proyecto.

  Las actvidades principales son:
  - identificar la confi del soft
  - control de conf del soft
  - contabilidad del estado de la conf
  - auditoria de la conf del soft
  - gestion y entrega de la version del soft

  ASpectos a tener en cuenta para esto:
-  ORganizacion y responsabilidades del SCM:
   identificar claramente los roles dentro del proyecto
- REcursos y horarios:
  identificar el personal, las herramientas y elaborar un horario (schedule) de las actividades
  e hitos.
- Seleccion e implementacion de herramientas:
  se denominan bancos de trabajo (puede ser abierto o integrado)
- Control de proveedores y subcontratistas:
  tomar en cuenta para la planificacion el software y mano de obra de terceros.
- Control de interfaz:
  especificar como se conectan los componenetes.
  Se lleva a cabo dentro del proceso y a nivel del sistema

#### Plan SCM

Los resutlados de la planificacion se poenen e un plan de gestion, un doc que sirve como
referencia para el proceso SCM.
Se actualiza segun sea necesario.

Los requisitos que debe contener son:
- introduccion (proposito, alcance y glosario) 
- Gestion SCMintroduccion (responsabilidades, organizacion, autoridades y procedimientos)
- Actividades SCM (lo que ya esta listado mas arriba)
- Cronogramas SCM
- Recursos SCM (herramientas, recursos fisicos y humanos)
- Mantenimiento

#### Vigilancia de la gestion de configuracion
 
Una vez se implementa el proceso del SCM, es necesario cierto nivel de vigilancia para ver que
se cumplan los requerimientos del SQA.
El responsable de SCM se encarga de monitorear a las personas encargadas de cada funcion

Para ello se requiere de herramientas y medidas de medicion de la calidad y estado del proceso
SCM

Asi tambien se pueden realizar auditorias durante el proceso SCM.

### Identificacion de la conf del soft

Identifcica los elementos a controlar y los esquemas de identificacion de elementos y
versiones, asi como las tecnicas y herramientas a ser utilizads.

Su desgloce:
- Identificar los elementos a controlar:
  identificar y categorizar los elementos del software a monitoriear:
    + Configuracion del software (engloba a las caracts funcionales y de hardw o softw)
    + Un elemento de la configuracion del software (conjunto de soft y hardw que se considera
      una sola unidad)
    + Relacion de los elemntos de la configuracion
    + Version del software (una instancia identificadda de un elemento de config)
    + Linea Base (version aprovada formalmente en un doc de conf de un elemento de la conf del
      software)
    + Adquirir elementos de la conf del softwar:
      se acoplan nuevos elementos en un instante de tiempo en base a una linea base
    + Libreria de software:
      coleccion controlada de soft y documentacion asociada

### Control de conf del softw

Gestionar los cambios durante la vida del softw.

Pasos:
- Solicitar, evaluar y aprobar cambios en el softw
    + Tablero de configraciuon de softw:
      la autoridad que acepta o rechaza los camibos propuesto se llama Junta de control de
      configuracion (CCB, SCCB si es solo responsable de softw) o en el lidel del proyectoa.
    + Proceso de solicitud de cambio del sftw
- Implementacion de los cambios:
  los camibos se implementan segun el cronograma y pueden ser sometidos a auditoria, etc. Se
  requiere un modo de rastrear cada cambio y su estado (tickets)
```txt
    [cambio] -- solicitud --> [revision]
    [revision] -- aprobacion --> [modificacion]
    [modificacion] -- acoplamiento --> [linea base]
    [implementacion] -- entrega --> [version de sftw]
```
- Desviacion y exenciones:
  Una desviación es una autorización escrita, otorgada antes de la fabricación de un
  artículo, para apartarse de un requisito de diseño o desempeño particular para un número
  específico de unidades o un período de tiempo específico.
  Una exención es una autorización por escrito para aceptar un elemento de configuración u
  otro elemento designado que, durante la producción o después de haber sido enviado para
  inspección, se desvía de los requisitos especificados pero que, sin embargo, se considera
  adecuado para su uso tal como está o después de ser reelaborado por un proveedor aprobado

### Contabilidad de la confi de softw

- Captura de informacion del estado de software (sacar estadisticas e informacion de las
  versiones de la confi)
- Generar informes (para respondeer a preguntas del management)

### Auditoria de la conf

es un examen independiente de un producto de trabajo o un conjunto de productos de trabajo para
evaluar el cumplimiento de especificaciones, estándares, acuerdos contractuales u otros
criterios.
se llevan a cabo de acuerdo con un proceso bien definido por parte del auditor

tipos:
- Auditoria de la configuracion funcional del sftw:
  garantizar que el softw cumpla con las especificaciones de la config
- Auditoria de la confi fisica del softw:
  de que el disenho y la documentacion sea consistente
- Auditoria de linea base:
  auditoria sobre elementos de referencia previamente muestreados (la linea base).
  En cristiano, no se hace sobre el estado actual de la configuracion, sino que sobre la linea
  base


### Gestion y entrega del softw

distribución de un elemento de configuración de software fuera la actividad de desarrollo.
Incluye lanzamientos internos y entregas al cliente.

Partes:
- Construccion del sftw:
  distribución de un elemento de configuración de software fuera la actividad de desarrollo
  (aka, como deployar en produccion gente y la doc asociada a eso).
  Es necesario que SCM tenga la capacidad de reproducir versiones anteriores con fines de
  recuperación, prueba, mantenimiento
  - Gestion de versiones:
    abarca la identificación, empaquetado y entrega de los elementos de un producto.
    Las notas de la versión suelen describir nuevas capacidades, problemas conocidos y
    requisitos de plataforma necesarios para el funcionamiento adecuado del producto
    (changelogs)


### HErramientas de gestion de la conf

Cuando se habla de herramientas de gestión de configuración de software, resulta útil
clasificarlas: 
- soporte individual, 
- soporte relacionado con proyectos
- soporte para procesos de toda la empresa


- Herramientas de control de versiones:
  rastrea, documenta y almacena elementos de configuración individuales, como código fuente y
  documentación externa.
- Construir herramientas de manejo:
  dichas herramientas compilan y vinculan una versión ejecutable del software.
  Las herramientas de creación más avanzadas producen diversos tipos de informes, entre otras
  tareas.
- Herramientas de control de cambios:
  apoyan principalmente el control de solicitudes de cambio y notificación de eventos (por
  ejemplo, cambios de estado de solicitudes de cambio, hitos alcanzados) (jira, sistema de
  tickets.


