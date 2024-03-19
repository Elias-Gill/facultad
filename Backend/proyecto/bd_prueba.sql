CREATE TABLE public.persona
(
    id_persona integer NOT NULL,

    -- colocar estos a7 dentro del java con anotaciones
    apellido character varying(50) NOT NULL,
    nombre character varying(50) NOT NULL,

    CONSTRAINT persona_pkey PRIMARY KEY (id_persona)
);
CREATE SEQUENCE public.persona_sec;

CREATE TABLE public.agenda
(
    id_agenda integer NOT NULL,
    fecha date NOT NULL,
    id_persona integer NOT NULL,
    actividad character varying(200) NOT NULL,

    -- clave primaria de la tabla
    CONSTRAINT agenda_pkey PRIMARY KEY (id_agenda),

    -- clave foranea que a la tabla persona con id_persona
    CONSTRAINT fk_1 FOREIGN KEY (id_persona)
        REFERENCES public.persona (id_persona) MATCH SIMPLE
        -- triggers para update o delete
        ON UPDATE NO ACTION
        ON DELETE NO ACTION
);
CREATE SEQUENCE public.agenda_sec;
