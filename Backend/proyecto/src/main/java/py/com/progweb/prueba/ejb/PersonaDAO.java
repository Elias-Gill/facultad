/* Los ejb son la representacion de nuestra logica de negocios
independiente a la arquitectura de la BD por debajo

Transforma las entidades que recibimos del ORM y hacemos algo usando el widfly
*/
package py.com.progweb.prueba.ejb;

import javax.ejb.Stateless;
import javax.persistence.EntityManager;
import javax.persistence.PersistenceContext;
import javax.persistence.Query;

import java.util.List;

import py.com.progweb.prueba.model.Persona;

@Stateless
public class PersonaDAO {
    /*
     * El entity manager que se va a encargar de manejar los objetos por nosotros.
     * La unidad de persistencia le dice al entorno (wildfly) a cual bd apuntar.
     * Ver en el persistence.xml el nombre de la unidad de persistencia
     */
    @PersistenceContext(unitName = "pruebaPU")
    private EntityManager em;

    public void agregarPersona(Persona entidad) {
        this.em.persist(entidad);
    }

    public List<Persona> listaPersonas() {
        Query q = this.em.createQuery("select p from Persona p");
        return (List<Persona>) q.getResultList();
    }
}
