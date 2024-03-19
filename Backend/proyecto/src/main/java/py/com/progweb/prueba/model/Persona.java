package py.com.progweb.prueba.model;

import javax.persistence.Basic;
import javax.persistence.Column;
import javax.persistence.Entity;
import javax.persistence.GeneratedValue;
import javax.persistence.GenerationType;
import javax.persistence.Id;
import javax.persistence.SequenceGenerator;
import javax.persistence.Table;

@Entity
@Table(name = "persona")
public class Persona {
    @Id // identificador de la tabla
    @Column(name = "id_persona") // nombre de la columna
    @Basic(optional = false) // si es un campo opcional en la insercion (obvio no)
    @GeneratedValue(generator = "personaSec", strategy = GenerationType.SEQUENCE) // nuevo generador de secuencia
    @SequenceGenerator(name = "personaSec", sequenceName = "persona_sec", allocationSize = 0) // asociar el generador a
                                                                                               // la secuencia
    private Integer idPersona;

    @Column(name = "nombre", length = 50)
    @Basic(optional = false)
    private String nombre;

    @Column(name = "apellido", length = 50)
    @Basic(optional = false)
    private String apellido;

    public Persona() {
    }

    // -- getters and setters --

    public Integer getIdPersona() {
        return idPersona;
    }

    public void setIdPersona(Integer idPersona) {
        this.idPersona = idPersona;
    }

    public String getNombre() {
        return nombre;
    }

    public void setNombre(String nombre) {
        this.nombre = nombre;
    }

    public String getApellido() {
        return apellido;
    }

    public void setApellido(String apellido) {
        this.apellido = apellido;
    }

}
